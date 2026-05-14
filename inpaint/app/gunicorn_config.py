import prometheus_client.multiprocess as prometheus_multiprocess


def child_exit(server, worker):
    prometheus_multiprocess.mark_process_dead(worker.pid)
