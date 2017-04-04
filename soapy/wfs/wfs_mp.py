
import multiprocessing

from .shackhartmannfast import ShackHartmannFast

class WFS_MP(object):
    def __init__(self, soapy_config, n_wfs=0, mask=None):

        self.input_queue = multiprocessing.Queue()
        self.output_queue = multiprocessing.Queue()
        self.exception_queue = multiprocessing.Queue()

        self.proc = multiprocessing.Process(
                target=self._run_process,
                args=(soapy_config, n_wfs, mask, self.input_queue, self.output_queue, self.exception_queue),
                daemon=True)

        self.proc.start()

        # Check if there's something gone wrong on init - raise exception if so
        exc = self.exception_queue.get()
        if exc is not None:
            raise exc

        self.n_measurements = self.output_queue.get()

    def _run_process(
            self, soapy_config, n_wfs, mask,
            input_queue, output_queue, exception_queue):

        try:
            wfs_obj = eval("{}".format(soapy_config.wfss[n_wfs].type))
            wfs = wfs_obj(soapy_config, n_wfs, mask)

            # Pass back some params that we need to know about the WFS
            output_queue.put(wfs.n_measurements)
            exception_queue.put(None)

        except Exception as exc:
            exception_queue.put(exc)

        running = True
        while running:

            input_data, input_kwargs = input_queue.get()

            if input_data is None:
                running = False
                break
            try:
                slopes = wfs.frame(*input_data, **input_kwargs)
                output_queue.put((slopes, wfs.detector, wfs.uncorrectedPhase))
                exception_queue.put(None)

            except Exception as exc:
                output_queue.put(None)
                exception_queue.put(exc)

    def start_frame(self, *args, **kwargs):

        self.input_queue.put((args, kwargs))


    def get_frame(self):


        self.slopes, self.wfsDetectorPlane, self.uncorrectedPhase = self.output_queue.get()

        exc = self.exception_queue.get()
        if exc is not None:
            raise exc

        return self.slopes

    def frame(self, *args, **kwargs):
        self.start_frame(*args, **kwargs)
        return self.get_frame()