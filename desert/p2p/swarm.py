"""Shared-mesh P2P layer for desert.

Every desert process joins the same GossipSub topic (``desert/swarm/v1``). A
separate **bootstrap server** runs a libp2p peer and serves HTTP
``GET /v1/bootstrap`` with dialable multiaddrs. All desert nodes set
``DESERT_BOOTSTRAP_URL`` and dial those seeds once at startup (libp2p
``BootstrapDiscovery``). After one successful hop, GossipSub propagates peer
records across the mesh.

The task protocol (``/desert/task/2.0.0``) lives in ``p2p.task_service``;
SwarmNode only handles presence.
"""

from __future__ import annotations

import inspect
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Callable

import multiaddr
import trio

from libp2p import new_host
from libp2p.abc import IHost, PeerInfo
from libp2p.custom_types import TProtocol
from libp2p.discovery.bootstrap.bootstrap import BootstrapDiscovery
from libp2p.discovery.events.peerDiscovery import peerDiscovery
from libp2p.peer.id import ID
from libp2p.peer.peerinfo import info_from_p2p_addr
from libp2p.peer.peerstore import PERMANENT_ADDR_TTL
from libp2p.pubsub.gossipsub import GossipSub, PROTOCOL_ID as GS_PROTOCOL_ID
from libp2p.pubsub.pubsub import Pubsub
from libp2p.tools.async_service import background_trio_service
from libp2p.utils.address_validation import find_free_port, get_available_interfaces

from p2p.bootstrap_client import fetch_bootstrap_multiaddrs

log = logging.getLogger(__name__)

SWARM_TOPIC = "desert/swarm/v1"

ANNOUNCE_EVERY_SEC = 5.0
PEER_TTL_SEC = 30.0

GS_DEGREE = 6
GS_DEGREE_LOW = 4
GS_DEGREE_HIGH = 12

_ROLES = frozenset({"orchestrator", "worker", "bootstrap"})


@dataclass
class PeerRecord:
    peer_id: str
    role: str
    multiaddrs: list[str]
    caps: list[str]
    worker_id: str | None = None
    last_seen: float = field(default_factory=time.monotonic)


class SwarmNode:
    """libp2p host + GossipSub presence table."""

    def __init__(
        self,
        *,
        role: str,
        caps: list[str] | None = None,
        worker_id: str | None = None,
        listen_port: int | None = None,
        announce_multiaddr: str | None = None,
        bootstrap_url: str | None = None,
        topic: str = SWARM_TOPIC,
    ) -> None:
        if role not in _ROLES:
            raise ValueError(f"role must be one of {_ROLES}, got {role!r}")
        self.role = role
        self.caps = list(caps or [])
        self.worker_id = worker_id
        self.topic = topic
        env_port = os.environ.get("DESERT_P2P_LISTEN_PORT")
        self.listen_port: int = listen_port or (int(env_port) if env_port else 0)
        self.announce_multiaddr = (
            announce_multiaddr
            if announce_multiaddr is not None
            else (os.environ.get("DESERT_P2P_ANNOUNCE_ADDR") or "").strip()
        )
        self._bootstrap_url = (bootstrap_url or "").strip() or None

        self.host: IHost | None = None
        self.pubsub: Pubsub | None = None
        self._table: dict[str, PeerRecord] = {}
        self._listeners: list[Callable[[PeerRecord], None]] = []

    @property
    def peer_id(self) -> str:
        assert self.host is not None, "SwarmNode not started"
        return self.host.get_id().to_string()

    def self_multiaddrs(self) -> list[str]:
        if self.host is None:
            return []
        pid = self.peer_id
        seen: list[str] = []
        if self.announce_multiaddr:
            a = self.announce_multiaddr.rstrip("/")
            if "/p2p/" not in a:
                a = f"{a}/p2p/{pid}"
            seen.append(a)
        for a in self.host.get_addrs():
            s = str(a)
            if s not in seen:
                seen.append(s)
        return seen

    def peers(self) -> list[PeerRecord]:
        now = time.monotonic()
        keep: dict[str, PeerRecord] = {}
        for pid, rec in self._table.items():
            if now - rec.last_seen <= PEER_TTL_SEC:
                keep[pid] = rec
        self._table = keep
        self_pid = self.peer_id if self.host else None
        out = [r for pid, r in keep.items() if pid != self_pid]
        out.sort(key=lambda r: r.last_seen, reverse=True)
        return out

    def peers_by_role(self, role: str) -> list[PeerRecord]:
        return [r for r in self.peers() if r.role == role]

    def on_peer(self, cb: Callable[[PeerRecord], None]) -> None:
        self._listeners.append(cb)

    async def run(
        self,
        nursery: trio.Nursery,
        setup: Callable[[IHost, "SwarmNode"], object] | None = None,
    ) -> None:
        self.host = new_host()
        port = self.listen_port or find_free_port()
        listen_addrs = get_available_interfaces(port)

        gossipsub = GossipSub(
            protocols=[TProtocol(GS_PROTOCOL_ID)],
            degree=GS_DEGREE,
            degree_low=GS_DEGREE_LOW,
            degree_high=GS_DEGREE_HIGH,
            time_to_live=60,
            heartbeat_interval=2,
            direct_connect_initial_delay=0.1,
            direct_connect_interval=60,
            do_px=True,
            max_messages_per_topic_per_second=200.0,
            eclipse_protection_enabled=False,
            spam_protection_enabled=False,
        )
        pubsub = Pubsub(self.host, router=gossipsub, strict_signing=False)
        self.pubsub = pubsub

        async with self.host.run(listen_addrs=listen_addrs):
            log.info(
                "swarm host=%s role=%s listening on %s",
                self.peer_id[:12] + "…",
                self.role,
                [str(a) for a in self.host.get_addrs()],
            )

            if setup is not None:
                maybe = setup(self.host, self)
                if maybe is not None and inspect.isawaitable(maybe):
                    await maybe  # type: ignore[misc]

            async with background_trio_service(gossipsub), background_trio_service(pubsub):
                await pubsub.wait_until_ready()
                subscription = await pubsub.subscribe(self.topic)

                peerDiscovery.register_peer_discovered_handler(self._on_peer_discovered_sync)

                swarm = self.host.get_network()
                # Bootstrap server is the rendezvous peer; it does not dial itself.
                if self.role != "bootstrap":
                    nursery.start_soon(self._run_bootstrap, swarm)

                nursery.start_soon(self._announce_loop)
                nursery.start_soon(self._listen_loop, subscription)

                await trio.sleep_forever()

    def _valid_seeds(self, addrs: list[str]) -> list[str]:
        good: list[str] = []
        for s in addrs:
            try:
                multiaddr.Multiaddr(s)
                good.append(s)
            except Exception as e:
                log.warning("ignoring bad seed %r: %s", s, e)
        return good

    async def _run_bootstrap(self, swarm) -> None:
        while True:
            url = self._bootstrap_url or (os.environ.get("DESERT_BOOTSTRAP_URL") or "").strip()
            if not url:
                log.warning("DESERT_BOOTSTRAP_URL is not set; retry in 2s")
                await trio.sleep(2)
                continue
            try:
                seeds = self._valid_seeds(
                    await trio.to_thread.run_sync(lambda: fetch_bootstrap_multiaddrs(url))
                )
            except Exception as e:
                log.warning("bootstrap HTTP %s: %s (retry in 2s)", url, e)
                await trio.sleep(2)
                continue
            if not seeds:
                log.warning("bootstrap returned no multiaddrs; retry in 2s")
                await trio.sleep(2)
                continue
            try:
                bd = BootstrapDiscovery(swarm, list(seeds))
                await bd.start()
                log.info("bootstrap dial finished (%d seed(s))", len(seeds))
                return
            except Exception as e:
                log.warning("BootstrapDiscovery failed: %s (retry in 5s)", e)
                await trio.sleep(5)

    def _on_peer_discovered_sync(self, peer_info: PeerInfo) -> None:
        if self.host is None:
            return
        if peer_info.peer_id == self.host.get_id():
            return
        try:
            trio.lowlevel.spawn_system_task(self._connect_peer, peer_info)
        except RuntimeError:
            pass

    async def _connect_peer(self, peer_info: PeerInfo) -> None:
        assert self.host is not None
        try:
            await self.host.connect(peer_info)
            log.info("swarm dialed peer=%s", str(peer_info.peer_id)[:12] + "…")
        except Exception as e:
            log.debug("swarm dial %s failed: %s", peer_info.peer_id, e)

    def _self_record_bytes(self) -> bytes:
        assert self.host is not None and self.pubsub is not None
        addrs: list[str] = []
        if self.announce_multiaddr:
            addrs.append(self.announce_multiaddr)
        for a in self.host.get_addrs():
            s = str(a)
            if s not in addrs:
                addrs.append(s)
        payload = {
            "peer_id": self.peer_id,
            "role": self.role,
            "multiaddrs": addrs,
            "caps": self.caps,
            "worker_id": self.worker_id,
            "ts": time.time(),
        }
        return json.dumps(payload, separators=(",", ":")).encode("utf-8")

    async def _announce_loop(self) -> None:
        assert self.pubsub is not None
        while True:
            try:
                await self.pubsub.publish(self.topic, self._self_record_bytes())
            except Exception as e:
                log.debug("announce publish failed: %s", e)
            await trio.sleep(ANNOUNCE_EVERY_SEC)

    async def _listen_loop(self, subscription) -> None:
        while True:
            msg = await subscription.get()
            try:
                rec = json.loads(msg.data.decode("utf-8"))
                pid = str(rec.get("peer_id") or "")
                role = str(rec.get("role") or "")
                if not pid or role not in _ROLES:
                    continue
                if self.host is not None and pid == self.peer_id:
                    continue
                record = PeerRecord(
                    peer_id=pid,
                    role=role,
                    multiaddrs=[str(m) for m in (rec.get("multiaddrs") or [])],
                    caps=[str(c) for c in (rec.get("caps") or [])],
                    worker_id=(str(rec["worker_id"]) if rec.get("worker_id") else None),
                    last_seen=time.monotonic(),
                )
                self._table[pid] = record
                for cb in list(self._listeners):
                    try:
                        cb(record)
                    except Exception:
                        log.exception("swarm peer callback")
                self._remember_addrs(record)
            except Exception as e:
                log.debug("swarm listen: bad record: %s", e)

    def _remember_addrs(self, rec: PeerRecord) -> None:
        if self.host is None:
            return
        peerstore = self.host.get_peerstore()
        try:
            pid = ID.from_base58(rec.peer_id)
        except Exception:
            return
        maddrs: list[multiaddr.Multiaddr] = []
        for s in rec.multiaddrs:
            try:
                m = multiaddr.Multiaddr(s)
            except Exception:
                continue
            if m.value_for_protocol("p2p") is None:
                try:
                    m = m.encapsulate(multiaddr.Multiaddr(f"/p2p/{rec.peer_id}"))
                except Exception:
                    pass
            maddrs.append(m)
        if not maddrs:
            return
        try:
            peerstore.add_addrs(pid, maddrs, PERMANENT_ADDR_TTL)
        except Exception:
            pass

    def peer_info(self, peer_id: str) -> PeerInfo | None:
        rec = self._table.get(peer_id)
        if rec is None:
            return None
        for s in rec.multiaddrs:
            try:
                m = multiaddr.Multiaddr(s)
                if m.value_for_protocol("p2p") is None:
                    m = m.encapsulate(multiaddr.Multiaddr(f"/p2p/{rec.peer_id}"))
                return info_from_p2p_addr(m)
            except Exception:
                continue
        return None
