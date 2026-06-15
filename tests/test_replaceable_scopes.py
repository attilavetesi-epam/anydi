"""Tests for replaceable scope support (reentry="replace")."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass

import pytest

from anydi import Container

from tests.fixtures import UniqueId


@dataclass
class ScopedValue:
    label: str


class DependentService:
    def __init__(self, value: ScopedValue) -> None:
        self.value = value


@pytest.fixture
def container() -> Container:
    return Container()


class TestReplaceableScopeBasic:
    """Tests for basic replace behavior with replaceable=True."""

    def test_replace_creates_new_context(self, container: Container) -> None:
        """Replacing creates a fresh InstanceContext each time."""
        container.register_scope("task", replaceable=True)
        container.register(str, lambda: "value", scope="task")

        with container.scoped_context("task") as ctx_outer:
            with container.scoped_context("task", reentry="replace") as ctx_inner:
                assert ctx_outer is not ctx_inner

    def test_replace_has_own_instance_cache(self, container: Container) -> None:
        """Each nested scope has its own cache — instances are not shared."""
        container.register_scope("task", replaceable=True)

        call_count = 0

        def make_str() -> str:
            nonlocal call_count
            call_count += 1
            return f"instance_{call_count}"

        container.register(str, make_str, scope="task")

        with container.scoped_context("task"):
            outer_val = container.resolve(str)
            assert outer_val == "instance_1"

            with container.scoped_context("task", reentry="replace"):
                inner_val = container.resolve(str)
                assert inner_val == "instance_2"
                assert inner_val != outer_val

    def test_replace_cleanup_does_not_affect_parent(self, container: Container) -> None:
        """Exiting a nested scope restores the parent scope's instances."""
        container.register_scope("task", replaceable=True)

        call_count = 0

        def make_str() -> str:
            nonlocal call_count
            call_count += 1
            return f"instance_{call_count}"

        container.register(str, make_str, scope="task")

        with container.scoped_context("task"):
            outer_val = container.resolve(str)
            assert outer_val == "instance_1"

            with container.scoped_context("task", reentry="replace"):
                inner_val = container.resolve(str)
                assert inner_val == "instance_2"

            restored_val = container.resolve(str)
            assert restored_val == outer_val

    def test_replace_singleton_dependency(self, container: Container) -> None:
        """Nested scopes can resolve singleton dependencies."""
        container.register_scope("task", replaceable=True)
        container.register(int, lambda: 42, scope="singleton")

        def make_str(x: int) -> str:
            return f"value: {x}"

        container.register(str, make_str, scope="task")

        with container:
            with container.scoped_context("task"):
                outer_val = container.resolve(str)
                assert outer_val == "value: 42"

                with container.scoped_context("task", reentry="replace"):
                    inner_val = container.resolve(str)
                    assert inner_val == "value: 42"

    def test_replace_three_levels(self, container: Container) -> None:
        """Three levels of nesting with from_context work correctly."""
        container.register_scope("task", replaceable=True)
        container.register(ScopedValue, scope="task", from_context=True)

        with container.scoped_context("task") as ctx1:
            ctx1.set(ScopedValue, ScopedValue("level-1"))
            assert container.resolve(ScopedValue).label == "level-1"

            with container.scoped_context("task", reentry="replace") as ctx2:
                ctx2.set(ScopedValue, ScopedValue("level-2"))
                assert container.resolve(ScopedValue).label == "level-2"

                with container.scoped_context("task", reentry="replace") as ctx3:
                    ctx3.set(ScopedValue, ScopedValue("level-3"))
                    assert container.resolve(ScopedValue).label == "level-3"

                assert container.resolve(ScopedValue).label == "level-2"

            assert container.resolve(ScopedValue).label == "level-1"

    def test_replace_resource_cleanup(self, container: Container) -> None:
        """Resources in nested scopes are cleaned up in correct order."""
        container.register_scope("task", replaceable=True)

        cleanup_log: list[str] = []

        @container.provider(scope="task")
        def provide_str() -> Iterator[str]:
            cleanup_log.append("start")
            yield "resource"
            cleanup_log.append("cleanup")

        with container.scoped_context("task"):
            container.resolve(str)

            with container.scoped_context("task", reentry="replace"):
                container.resolve(str)

            assert cleanup_log == ["start", "start", "cleanup"]

        assert cleanup_log == ["start", "start", "cleanup", "cleanup"]

    def test_replace_instance_caching_within_level(self, container: Container) -> None:
        """Instances are cached within each nesting level."""
        container.register_scope("task", replaceable=True)
        container.register(UniqueId, scope="task")

        with container.scoped_context("task"):
            id1 = container.resolve(UniqueId)
            id2 = container.resolve(UniqueId)
            assert id1 is id2

            with container.scoped_context("task", reentry="replace"):
                id3 = container.resolve(UniqueId)
                id4 = container.resolve(UniqueId)
                assert id3 is id4
                assert id3 is not id1


class TestReplaceableScopeAsync:
    """Async replace behavior tests."""

    async def test_async_replace_creates_new_context(
        self, container: Container
    ) -> None:
        """Async replace creates a fresh InstanceContext each time."""
        container.register_scope("task", replaceable=True)
        container.register(str, lambda: "value", scope="task")

        async with container.ascoped_context("task") as ctx_outer:
            async with container.ascoped_context(
                "task", reentry="replace"
            ) as ctx_inner:
                assert ctx_outer is not ctx_inner

    async def test_async_replace_has_own_cache(self, container: Container) -> None:
        """Async nested scopes have isolated from_context values."""
        container.register_scope("task", replaceable=True)
        container.register(ScopedValue, scope="task", from_context=True)

        async with container.ascoped_context("task") as ctx1:
            ctx1.set(ScopedValue, ScopedValue("outer"))
            assert (await container.aresolve(ScopedValue)).label == "outer"

            async with container.ascoped_context("task", reentry="replace") as ctx2:
                ctx2.set(ScopedValue, ScopedValue("inner"))
                assert (await container.aresolve(ScopedValue)).label == "inner"

            assert (await container.aresolve(ScopedValue)).label == "outer"

    async def test_async_replace_resource_cleanup(self, container: Container) -> None:
        """Async resources in nested scopes are cleaned up in correct order."""
        container.register_scope("task", replaceable=True)

        cleanup_log: list[str] = []

        @container.provider(scope="task")
        async def provide_str() -> AsyncIterator[str]:
            cleanup_log.append("start")
            yield "resource"
            cleanup_log.append("cleanup")

        async with container.ascoped_context("task"):
            await container.aresolve(str)

            async with container.ascoped_context("task", reentry="replace"):
                await container.aresolve(str)

            assert cleanup_log == ["start", "start", "cleanup"]

        assert cleanup_log == ["start", "start", "cleanup", "cleanup"]


class TestReentryPolicy:
    """Tests for the reentry policy: reuse / replace / forbid."""

    def test_reentry_reuses_context_by_default(self, container: Container) -> None:
        """Default reentry='reuse' reuses the existing context."""
        container.register_scope("task")
        container.register(str, lambda: "value", scope="task")

        with container.scoped_context("task") as ctx1:
            with container.scoped_context("task") as ctx2:
                assert ctx1 is ctx2

    def test_reentry_raises_when_forbidden(self, container: Container) -> None:
        """reentry='forbid' raises RuntimeError when the scope is active."""
        container.register_scope("task")

        with container.scoped_context("task"):
            with pytest.raises(
                RuntimeError,
                match=r"re-entry is forbidden",
            ):
                with container.scoped_context("task", reentry="forbid"):
                    pass

    async def test_async_reentry_raises_when_forbidden(
        self, container: Container
    ) -> None:
        """Async reentry='forbid' raises RuntimeError when the scope is active."""
        container.register_scope("task")

        async with container.ascoped_context("task"):
            with pytest.raises(
                RuntimeError,
                match=r"re-entry is forbidden",
            ):
                async with container.ascoped_context("task", reentry="forbid"):
                    pass

    def test_forbid_raises_on_replaceable_scope(self, container: Container) -> None:
        """reentry='forbid' raises even on a replaceable scope."""
        container.register_scope("task", replaceable=True)

        with container.scoped_context("task"):
            with pytest.raises(
                RuntimeError,
                match=r"re-entry is forbidden",
            ):
                with container.scoped_context("task", reentry="forbid"):
                    pass

    async def test_async_forbid_raises_on_replaceable_scope(
        self, container: Container
    ) -> None:
        """Async reentry='forbid' raises even on a replaceable scope."""
        container.register_scope("task", replaceable=True)

        async with container.ascoped_context("task"):
            with pytest.raises(
                RuntimeError,
                match=r"re-entry is forbidden",
            ):
                async with container.ascoped_context("task", reentry="forbid"):
                    pass

    def test_replace_raises_on_non_replaceable_scope(
        self, container: Container
    ) -> None:
        """reentry='replace' on a non-replaceable active scope raises RuntimeError."""
        container.register_scope("task")

        with container.scoped_context("task"):
            with pytest.raises(
                RuntimeError,
                match=r"not replaceable",
            ):
                with container.scoped_context("task", reentry="replace"):
                    pass

    async def test_async_replace_raises_on_non_replaceable_scope(
        self, container: Container
    ) -> None:
        """Async reentry='replace' on a non-replaceable active scope raises."""
        container.register_scope("task")

        async with container.ascoped_context("task"):
            with pytest.raises(
                RuntimeError,
                match=r"not replaceable",
            ):
                async with container.ascoped_context("task", reentry="replace"):
                    pass

    def test_replace_on_first_entry_does_not_require_optin(
        self, container: Container
    ) -> None:
        """reentry='replace' on a first entry just creates a context (no re-entry)."""
        container.register_scope("task")

        with container.scoped_context("task", reentry="replace") as ctx:
            assert container.get_scoped_context("task") is ctx

    def test_builtin_request_scope_reentry_reuses(self, container: Container) -> None:
        """Request scope reentry reuses context by default."""
        with container.request_context() as ctx1:
            with container.request_context() as ctx2:
                assert ctx1 is ctx2


class TestGetScopedContext:
    """Tests for the public get_scoped_context() method."""

    def test_get_scoped_context_returns_current(self, container: Container) -> None:
        """Returns the active context for the specified scope."""
        container.register_scope("task", replaceable=True)

        with container.scoped_context("task") as ctx:
            assert container.get_scoped_context("task") is ctx

    def test_get_scoped_context_returns_innermost(self, container: Container) -> None:
        """Returns the innermost context when nested."""
        container.register_scope("task", replaceable=True)

        with container.scoped_context("task") as ctx_outer:
            assert container.get_scoped_context("task") is ctx_outer

            with container.scoped_context("task", reentry="replace") as ctx_inner:
                assert container.get_scoped_context("task") is ctx_inner

            assert container.get_scoped_context("task") is ctx_outer

    def test_get_scoped_context_raises_when_not_started(
        self, container: Container
    ) -> None:
        """Raises LookupError when scope is not active."""
        container.register_scope("task", replaceable=True)

        with pytest.raises(
            LookupError,
            match=r"The task context has not been started",
        ):
            container.get_scoped_context("task")

    def test_get_scoped_context_reserved_scope_raises(
        self, container: Container
    ) -> None:
        """Raises ValueError for reserved scopes."""
        with pytest.raises(ValueError, match=r"reserved scope"):
            container.get_scoped_context("singleton")

    def test_get_scoped_context_unregistered_scope_raises(
        self, container: Container
    ) -> None:
        """Raises ValueError for unregistered scopes."""
        with pytest.raises(ValueError, match=r"not registered scope"):
            container.get_scoped_context("unknown")


class TestReplaceableRegistration:
    """Tests for replaceable registration rules."""

    def test_replaceable_rejected_for_reserved_scopes(
        self, container: Container
    ) -> None:
        """Builtin scopes cannot be registered as replaceable."""
        with pytest.raises(ValueError, match=r"reserved"):
            container.register_scope("singleton", replaceable=True)

        with pytest.raises(ValueError, match=r"reserved"):
            container.register_scope("transient", replaceable=True)

    def test_has_scope_with_replaceable(self, container: Container) -> None:
        """has_scope returns True for replaceable scopes."""
        container.register_scope("task", replaceable=True)
        assert container.has_scope("task")


class TestReplaceableScopeWithBuild:
    """Tests that build() validation and compiled resolvers work with replace."""

    def test_build_validates_replaceable_scope(self, container: Container) -> None:
        """build() succeeds with replaceable scopes."""
        container.register_scope("task", replaceable=True)
        container.register(ScopedValue, scope="task", from_context=True)
        container.register(DependentService, scope="task")

        container.build()
        assert container.ready

    def test_build_then_replace_resolution(self, container: Container) -> None:
        """Nested resolution works after build() has compiled resolvers."""
        container.register_scope("task", replaceable=True)
        container.register(int, lambda: 99, scope="singleton")
        container.register(ScopedValue, scope="task", from_context=True)
        container.register(DependentService, scope="task")

        container.build()

        with container:
            with container.scoped_context("task") as ctx_a:
                ctx_a.set(ScopedValue, ScopedValue("outer"))
                svc_a = container.resolve(DependentService)
                singleton_a = container.resolve(int)

                with container.scoped_context("task", reentry="replace") as ctx_b:
                    ctx_b.set(ScopedValue, ScopedValue("inner"))
                    svc_b = container.resolve(DependentService)
                    singleton_b = container.resolve(int)

                    assert svc_a is not svc_b
                    assert svc_b.value.label == "inner"
                    assert singleton_a is singleton_b

                assert container.resolve(DependentService) is svc_a
                assert container.resolve(ScopedValue).label == "outer"

    async def test_build_then_async_replace_resolution(
        self, container: Container
    ) -> None:
        """Async nested resolution works after build()."""
        container.register_scope("task", replaceable=True)
        container.register(ScopedValue, scope="task", from_context=True)
        container.register(DependentService, scope="task")

        container.build()

        async with container.ascoped_context("task") as ctx_a:
            ctx_a.set(ScopedValue, ScopedValue("outer"))
            svc_a = await container.aresolve(DependentService)

            async with container.ascoped_context("task", reentry="replace") as ctx_b:
                ctx_b.set(ScopedValue, ScopedValue("inner"))
                svc_b = await container.aresolve(DependentService)

                assert svc_a is not svc_b
                assert svc_b.value.label == "inner"

            assert (await container.aresolve(ScopedValue)).label == "outer"
            assert (await container.aresolve(DependentService)) is svc_a


class TestReplaceableScopeChain:
    """End-to-end nested scope chain tests."""

    def test_replace_chain(self, container: Container) -> None:
        """Three-level nesting with context restoration on exit."""
        container.register_scope("task", replaceable=True)
        container.register(ScopedValue, scope="task", from_context=True)
        container.register(DependentService, scope="task")

        log: list[str] = []

        with container.scoped_context("task") as ctx1:
            ctx1.set(ScopedValue, ScopedValue("L1"))
            svc1 = container.resolve(DependentService)
            log.append(f"start: {svc1.value.label}")

            with container.scoped_context("task", reentry="replace") as ctx2:
                ctx2.set(ScopedValue, ScopedValue("L2"))
                svc2 = container.resolve(DependentService)
                log.append(f"nested: {svc2.value.label}")

                with container.scoped_context("task", reentry="replace") as ctx3:
                    ctx3.set(ScopedValue, ScopedValue("L3"))
                    svc3 = container.resolve(DependentService)
                    log.append(f"nested: {svc3.value.label}")

                assert container.resolve(ScopedValue).label == "L2"
                log.append(f"resumed: {svc2.value.label}")

            assert container.resolve(ScopedValue).label == "L1"
            log.append(f"resumed: {svc1.value.label}")

        assert log == [
            "start: L1",
            "nested: L2",
            "nested: L3",
            "resumed: L2",
            "resumed: L1",
        ]

    async def test_async_replace_chain(self, container: Container) -> None:
        """Async three-level nesting with context restoration."""
        container.register_scope("task", replaceable=True)
        container.register(ScopedValue, scope="task", from_context=True)

        async def make_service(value: ScopedValue) -> DependentService:
            return DependentService(value)

        container.register(DependentService, make_service, scope="task")

        async with container.ascoped_context("task") as ctx1:
            ctx1.set(ScopedValue, ScopedValue("L1"))
            svc1 = await container.aresolve(DependentService)
            assert svc1.value.label == "L1"

            async with container.ascoped_context("task", reentry="replace") as ctx2:
                ctx2.set(ScopedValue, ScopedValue("L2"))
                svc2 = await container.aresolve(DependentService)
                assert svc2.value.label == "L2"

            assert (await container.aresolve(ScopedValue)).label == "L1"


class TestReplaceableScopeParallelAsync:
    """Tests for concurrent replace via asyncio.gather / create_task."""

    async def test_parallel_gather_isolated_contexts(
        self, container: Container
    ) -> None:
        """Concurrent scopes via gather don't cross-contaminate."""
        container.register_scope("task", replaceable=True)
        container.register(ScopedValue, scope="task", from_context=True)
        container.register(DependentService, scope="task")

        async def run_task(label: str) -> tuple[str, str]:
            async with container.ascoped_context("task") as ctx:
                ctx.set(ScopedValue, ScopedValue(label))
                svc = await container.aresolve(DependentService)
                await asyncio.sleep(0)
                after = await container.aresolve(ScopedValue)
                return svc.value.label, after.label

        results = await asyncio.gather(
            run_task("A"),
            run_task("B"),
            run_task("C"),
        )

        assert results == [("A", "A"), ("B", "B"), ("C", "C")]

    async def test_parallel_gather_with_parent_scope(
        self, container: Container
    ) -> None:
        """Parallel replaced scopes inside a parent scope don't affect the parent."""
        container.register_scope("task", replaceable=True)
        container.register(ScopedValue, scope="task", from_context=True)

        async def run_task(label: str) -> str:
            async with container.ascoped_context("task", reentry="replace") as ctx:
                ctx.set(ScopedValue, ScopedValue(label))
                await asyncio.sleep(0)
                return (await container.aresolve(ScopedValue)).label

        async with container.ascoped_context("task") as parent_ctx:
            parent_ctx.set(ScopedValue, ScopedValue("parent"))

            results = await asyncio.gather(
                run_task("A"),
                run_task("B"),
                run_task("C"),
            )

            assert results == ["A", "B", "C"]
            assert (await container.aresolve(ScopedValue)).label == "parent"

    async def test_parallel_create_task_isolated_contexts(
        self, container: Container
    ) -> None:
        """Concurrent replaced scopes via create_task don't cross-contaminate."""
        container.register_scope("task", replaceable=True)
        container.register(ScopedValue, scope="task", from_context=True)

        results: dict[str, str] = {}

        async def run_task(label: str) -> None:
            async with container.ascoped_context("task", reentry="replace") as ctx:
                ctx.set(ScopedValue, ScopedValue(label))
                await asyncio.sleep(0)
                resolved = await container.aresolve(ScopedValue)
                results[label] = resolved.label

        async with container.ascoped_context("task") as parent_ctx:
            parent_ctx.set(ScopedValue, ScopedValue("parent"))

            tasks = [asyncio.create_task(run_task(f"T{i}")) for i in range(5)]
            await asyncio.gather(*tasks)

            for i in range(5):
                assert results[f"T{i}"] == f"T{i}"

            assert (await container.aresolve(ScopedValue)).label == "parent"

    async def test_parallel_replace_with_build(self, container: Container) -> None:
        """Parallel replaced scopes work correctly after build()."""
        container.register_scope("task", replaceable=True)
        container.register(int, lambda: 42, scope="singleton")
        container.register(ScopedValue, scope="task", from_context=True)
        container.register(DependentService, scope="task")

        container.build()

        async def run_task(label: str) -> tuple[str, int]:
            async with container.ascoped_context("task", reentry="replace") as ctx:
                ctx.set(ScopedValue, ScopedValue(label))
                await asyncio.sleep(0)
                svc = await container.aresolve(DependentService)
                singleton_val = await container.aresolve(int)
                return svc.value.label, singleton_val

        async with container:
            async with container.ascoped_context("task") as parent_ctx:
                parent_ctx.set(ScopedValue, ScopedValue("parent"))

                results = await asyncio.gather(
                    run_task("A"),
                    run_task("B"),
                    run_task("C"),
                )

                assert [r[0] for r in results] == ["A", "B", "C"]
                assert all(r[1] == 42 for r in results)

                assert (await container.aresolve(ScopedValue)).label == "parent"

    def test_parallel_threads_isolated_contexts(self, container: Container) -> None:
        """Concurrent scopes across threads are fully isolated."""
        container.register_scope("task", replaceable=True)
        container.register(UniqueId, scope="task")

        results: list[tuple[UniqueId, UniqueId]] = []

        def worker() -> None:
            with container.scoped_context("task"):
                id1 = container.resolve(UniqueId)
                id2 = container.resolve(UniqueId)
                results.append((id1, id2))

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        for id1, id2 in results:
            assert id1 is id2

        unique_ids = {id(pair[0]) for pair in results}
        assert len(unique_ids) == 5


class TestReplaceableScopeWithTestMode:
    """Tests for replaced scopes combined with test_mode / override."""

    def test_replace_preserves_outer_deps_in_test_mode(
        self, container: Container
    ) -> None:
        """Outer service attrs still resolve to outer deps inside an inner scope."""
        container.enable_test_mode()
        container.register_scope("task", replaceable=True)
        container.register(ScopedValue, scope="task", from_context=True)
        container.register(DependentService, scope="task")

        with container.scoped_context("task") as ctx_a:
            ctx_a.set(ScopedValue, ScopedValue("A"))
            svc_a = container.resolve(DependentService)
            assert svc_a.value.label == "A"

            with container.scoped_context("task", reentry="replace") as ctx_b:
                ctx_b.set(ScopedValue, ScopedValue("B"))
                svc_b = container.resolve(DependentService)
                assert svc_b.value.label == "B"

                # While in scope B, outer service must still see scope A's value
                assert svc_a.value.label == "A"

            assert svc_a.value.label == "A"

    def test_replace_override_in_test_mode(self, container: Container) -> None:
        """Override works correctly within a nested scope."""
        container.enable_test_mode()
        container.register_scope("task", replaceable=True)
        container.register(ScopedValue, scope="task", from_context=True)
        container.register(DependentService, scope="task")

        with container.scoped_context("task") as ctx:
            ctx.set(ScopedValue, ScopedValue("real"))
            svc = container.resolve(DependentService)
            assert svc.value.label == "real"

            mock_val = ScopedValue("mocked")
            with container.override(ScopedValue, mock_val):
                assert svc.value.label == "mocked"

            assert svc.value.label == "real"

    def test_replace_three_levels_test_mode(self, container: Container) -> None:
        """Three levels of nesting in test_mode preserve per-level deps."""
        container.enable_test_mode()
        container.register_scope("task", replaceable=True)
        container.register(ScopedValue, scope="task", from_context=True)
        container.register(DependentService, scope="task")

        with container.scoped_context("task") as ctx1:
            ctx1.set(ScopedValue, ScopedValue("L1"))
            svc1 = container.resolve(DependentService)

            with container.scoped_context("task", reentry="replace") as ctx2:
                ctx2.set(ScopedValue, ScopedValue("L2"))
                svc2 = container.resolve(DependentService)

                with container.scoped_context("task", reentry="replace") as ctx3:
                    ctx3.set(ScopedValue, ScopedValue("L3"))
                    svc3 = container.resolve(DependentService)
                    assert svc3.value.label == "L3"
                    assert svc2.value.label == "L2"
                    assert svc1.value.label == "L1"

                assert svc2.value.label == "L2"
                assert svc1.value.label == "L1"

            assert svc1.value.label == "L1"
