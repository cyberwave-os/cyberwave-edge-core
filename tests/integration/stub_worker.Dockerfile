# Minimal stub image used by the worker-lifecycle real-Docker integration
# tests (CYB-2139). Edge Core's ``_run_container`` passes no CMD/entrypoint,
# so the image's default CMD must be long-running on its own. ``alpine:3.20``
# alone exits immediately when ``docker run --detach`` is used; this layer
# adds ``sleep infinity`` so the container reaches the ``running`` state and
# stays there until the test stops it.
FROM alpine:3.20
CMD ["sleep", "infinity"]
