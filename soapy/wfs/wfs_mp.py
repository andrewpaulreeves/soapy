
import multiprocessing

from . import *

class ShackHartmannMP(object):
    def __init__(self, soapy_config, n_wfs=0, mask=None):

        self.input_queue = multiprocessing.Queue()
        self.output_queue = multiprocessing.Queue()
        self.proc = multiprocessing.Process(
                target=self._run_process,
                args=(soapy_config, n_wfs, mask, self.input_queue, self.output_queue)
                )

    def _run_process(
            self, soapy_config, n_wfs, mask,
            input_queue, output_queue):

        wfs = eval("{}({}, {}, {})".format(
                soapy_config.wfss[n_wfs].type,
                soapy_config, n_wfs, mask))

        running = True
        while running:

            input_data = input_queue.get()

            if input_data == None:
                running = False
                break

            slopes = wfs.frame(*input_data)

            output_queue.put(slopes)


    def frame(self, *args):

        self.input_queue.put(args)

        self.slopes = self.output_queue.get()

        return