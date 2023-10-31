"""
python helper functions
"""

from contextlib import contextmanager
import time
import sys

this = sys.modules[__name__]

# we can explicitly make assignments on it 
this.is_first_time=True

@contextmanager
def timing(msg, verbose=True):
    if type(verbose) is tuple and len(verbose)==3 and verbose[0]:
        output_folder=verbose[1]
        filename_suffix=verbose[2]
        print("\tRecording [Timer] %s started... to %s" % (msg,filename_suffix))
        tstart = time.time()
        yield
        duration = time.time() - tstart
        print("\t[Timer] %s done in %.3f seconds" % (msg, duration))
        print("\t---------------------------------")
        with open(f"{output_folder}/rmp_timing.txt", "a") as myfile:
            if(this.is_first_time):
                myfile.write(f"\n{filename_suffix},{duration}")
                this.is_first_time=False
            else:
                myfile.write(f",{duration}")

    elif verbose:
        # print("\t---------------------------------")
        print("\t[Timer] %s started..." % (msg))
        tstart = time.time()
        yield
        print("\t[Timer] %s done in %.3f seconds" % (msg, time.time() - tstart))
        print("\t---------------------------------")
    else:
        yield

def merge_dicts(original, new_dict):
    if new_dict is not None:
        updated_dict = original.copy()
        updated_dict.update(new_dict)
    else:
        updated_dict = original
    return updated_dict
