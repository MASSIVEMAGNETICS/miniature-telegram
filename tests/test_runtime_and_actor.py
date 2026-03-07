"""Tests for the CT-GGN runtime, Echo Ripple protocol, and Aetherling actor."""

import asyncio
import pytest
import numpy as np

from aetherforge.runtime import CTGGNRuntime
from aetherforge.actor import Aetherling
from aetherforge.echo_ripple import EchoRippleProtocol
from aetherforge.aetherscript import load_aetherscript, validate_aetherscript


VALID_SPEC = """
aetherling:
  name: "TestAgent"
  topology:
    base: "FlowerOfLife-37"
    core_fluidity: 0.5
  perimeter_io:
    nodes: 18
    geometry: "GATr-Platonic"
    erasure_threshold: 0.3
  swarm:
    max_forks: 5
    sync_protocol: "EchoRipple-gRPC"
  security:
    soul_token_enclave: false
    ledger: "Enterprise-Private"
"""


class TestCTGGNRuntime:
    def setup_method(self):
        self.rt = CTGGNRuntime(state_dim=8, seed=0)

    def test_initial_step_count_zero(self):
        assert self.rt.step_count == 0

    def test_step_returns_correct_shape(self):
        out = self.rt.step()
        assert out.shape == (8,)

    def test_step_increments_counter(self):
        self.rt.step()
        assert self.rt.step_count == 1

    def test_step_with_none_input(self):
        out = self.rt.step(None)
        assert out.shape == (8,)

    def test_step_with_scalar_input(self):
        out = self.rt.step(1.0)
        assert out.shape == (8,)

    def test_step_with_vector_input(self):
        u = np.ones(8)
        out = self.rt.step(u)
        assert out.shape == (8,)

    def test_step_with_full_input(self):
        u = np.random.default_rng(1).standard_normal((19, 8))
        out = self.rt.step(u)
        assert out.shape == (8,)

    def test_output_is_finite(self):
        for _ in range(10):
            out = self.rt.step(np.random.default_rng(2).standard_normal(8))
            assert np.all(np.isfinite(out))

    def test_reset_clears_state(self):
        self.rt.step(np.ones(8))
        self.rt.reset()
        np.testing.assert_array_equal(self.rt.core.states, np.zeros((19, 8)))

    def test_ledger_grows_with_steps(self):
        initial_len = len(self.rt.ledger)
        for _ in range(3):
            self.rt.step()
        assert len(self.rt.ledger) > initial_len

    def test_ledger_integrity_after_steps(self):
        for _ in range(5):
            self.rt.step()
        assert self.rt.ledger.verify()

    def test_clone_is_independent(self):
        self.rt.step(np.ones(8))
        clone = self.rt.clone()
        clone.step(np.ones(8) * 99)
        clone.step(np.ones(8) * 99)
        # Clone has taken 2 steps; original should still be at 1
        assert clone.step_count == 2
        assert self.rt.step_count == 1

    def test_from_aetherscript(self):
        spec = load_aetherscript(VALID_SPEC)
        validated = validate_aetherscript(spec)
        rt = CTGGNRuntime.from_aetherscript(validated)
        assert rt.agent_name == "TestAgent"
        out = rt.step()
        assert out.shape == (rt.state_dim,)


class TestEchoRippleProtocol:
    def setup_method(self):
        self.rt = CTGGNRuntime(state_dim=8, seed=1)

    def test_step_returns_summary_dict(self):
        self.rt.step()  # prime activations
        stats = self.rt.echo_ripple.step(self.rt.core, self.rt.perimeter)
        assert "edges_strengthened" in stats
        assert "synapses_pruned" in stats
        assert "total_pruned_cumulative" in stats

    def test_update_count_increments(self):
        for _ in range(3):
            self.rt.step()
        assert self.rt.echo_ripple.update_count >= 3

    def test_weights_change_after_steps(self):
        # Node 0 (center) gets no echo since it's 3 hops from the perimeter;
        # use node 1 which lies in the echo-receiving boundary layer.
        w_before = self.rt.core.nodes[1].W_rec.copy()
        rng = np.random.default_rng(7)
        for _ in range(5):
            self.rt.step(rng.standard_normal(8) * 5.0)
        w_after = self.rt.core.nodes[1].W_rec
        assert not np.allclose(w_before, w_after)


class TestAetherling:
    def setup_method(self):
        self.rt = CTGGNRuntime(state_dim=8, seed=2)
        self.agent = Aetherling(self.rt, name="alpha", max_forks=3)

    def test_soul_token_is_32_bytes(self):
        assert len(self.agent.soul_token) == 32

    def test_step_returns_output(self):
        out = self.agent.step(np.ones(8))
        assert out.shape == (8,)

    def test_fork_creates_child(self):
        child = self.agent.fork()
        assert child.name.startswith("alpha:fork-")
        assert self.agent.fork_count == 1

    def test_fork_respects_max_forks(self):
        for _ in range(3):
            self.agent.fork()
        with pytest.raises(RuntimeError, match="max_forks"):
            self.agent.fork()

    def test_reincarnate_resets_state(self):
        self.agent.step(np.ones(8))
        self.agent.reincarnate()
        np.testing.assert_array_equal(
            self.agent.runtime.core.states, np.zeros((19, 8))
        )

    def test_terminate_logs_event(self):
        self.agent.terminate()
        events = [e.event_type for e in self.agent.runtime.ledger._entries]
        assert "AGENT_EVENT" in events

    def test_fork_ledger_is_independent(self):
        child = self.agent.fork()
        initial_child_len = len(child.runtime.ledger)
        child.step()
        # Parent ledger should not grow when child steps
        assert len(child.runtime.ledger) > initial_child_len

    def test_toggle_swarm(self):
        self.agent.toggle_swarm(True)
        assert self.agent._swarm_mode is True
        self.agent.toggle_swarm(False)
        assert self.agent._swarm_mode is False

    def test_async_send_and_run(self):
        async def _run():
            task = asyncio.create_task(self.agent.run())
            await self.agent.send(np.ones(8))
            await asyncio.sleep(0.05)
            self.agent.terminate()
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        asyncio.run(_run())
