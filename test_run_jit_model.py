import torch
from torchvision import models
from test_jit import get_execution_plan
from common_utils import enable_profiling_mode
import sys
import argparse
import logging
import time
from common_utils import freeze_rng_state

def test_allclose(a, b):
    if isinstance(a, (tuple, list)):
        if (not isinstance(b, (tuple, list))):
            logging.info("b isn't a tuple")
        if (len(a) != len(b)):
            logging.info("lengths aren't equal")
        assert(isinstance(b, (tuple, list)) and len(a) == len(b))

        for i in range(0, len(a)):
            if not test_allclose(a[i], b[i]):
                return False
    elif isinstance(a, torch.Tensor):
        return torch.allclose(a, b, 1e-04, 1e-04)
    elif isinstance(a, (dict)):
        assert(isinstance(b, dict))
        for k in a.keys():
            if not test_allclose(a[k], b[k]):
                return False
    elif isinstance(a, float):
        if (not isinstance(b, float)):
            logging.info('b isn\'t a float')
        assert(isinstance(b, float))
        return torch.allclose(torch.tensor([a]), torch.tensor([b]), 1e-04, 1e-04)
    elif isinstance(a, str):
        if (not isinstance(b, str)):
            logging.info('b isn\'t a string')
        assert(isinstance(b, str))
        return a == b
    else:
        logging.info("hit unexpected %s", str(type(a)))
        logging.info("dict = %s", str(a))
        raise AssertionError("Unexpected type {}".format(str(type(a))))
    return True

def get_plan(model):
    state = model._c.get_debug_state()
    plan = get_execution_plan(state)
    num_bailouts = plan.code.num_bailouts()
    return plan

def do_legacy_test(model, print_diff):
    torch._C._jit_set_profiling_executor(False)
    torch._C._jit_set_profiling_mode(False)
    logging.basicConfig(filename='jit_' + model.replace('/','-') + str(int(time.time())) + '.log', filemode='w', level=logging.DEBUG)
    logging.info("loading %s", model)
    jm = torch.jit.load(model)
    logging.info("evaling %s", model)
    jm.eval()
    logging.info("running legacy %s", model)
    with freeze_rng_state():
        po = jm()
    logging.info("running legacy %s", model)
    with freeze_rng_state():
        po2 = jm()
    if not test_allclose(po, po2):
        logging.error("legacy and legacy2 outputs aren't equal")
        if (print_diff):
            logging.error("po : %s", str(po))
            logging.error("po2 : %s", str(po2))

def do_test(model, bailout, print_diff):
    logging.basicConfig(filename='jit_' + model.replace('/','-') + str(int(time.time())) + '.log', filemode='w', level=logging.DEBUG)
    with enable_profiling_mode():
        logging.info("loading profiled %s", model)
        jm = torch.jit.load(model)
        #jm.eval()
        logging.info("running profiled %s", model)
        with freeze_rng_state():
            po = jm()
        logging.info("running profiled2 %s", model)
        with freeze_rng_state():
            po2 = jm()
        if not test_allclose(po, po2):
            logging.error("profiled and profiled2 outputs aren't equal")
            if (print_diff):
                logging.error("po : %s", str(po))
                logging.error("po2 : %s", str(po2))
        logging.info("running optimized %s", model)
        with freeze_rng_state():
            jo = jm()
        if not test_allclose(po, jo):
            logging.error("profiled and optimized outputs aren't equal")
            if (print_diff):
                logging.error("po : %s", str(po))
                logging.error("jo : %s", str(jo))
        plan = get_plan(jm)
        num_bailouts = plan.code.num_bailouts()
        logging.info("number of bailouts: %d", num_bailouts)
        if bailout:
            logging.info("triggering bailout %d ", bailout)
            plan.code.request_bailout(bailout)
            with freeze_rng_state():
                bo = jm()
            if not test_allclose(bo, jo):
                logging.error("bailout %d and optimized outputs aren't equal", bailout)
                if (print_diff):
                    logging.error("bo : %s", str(bo))
                    logging.error("jo : %s", str(jo))
        else:
            for i in range(0, num_bailouts):
                logging.info("triggering bailout %d ", i)
                plan.code.request_bailout(i)
                with freeze_rng_state():
                    bo = jm()
                if not test_allclose(bo, jo):
                    logging.error("bailout %d and optimized outputs aren't equal", i)
                    if (print_diff):
                        logging.error("bo : %s", str(bo))
                        logging.error("jo : %s", str(jo))
            #open("BAILOUTS_PASSED_jit_{}".format(model.replace('/', '-')), 'a').close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_name', nargs=1)
    parser.add_argument('--bailout', type=int)
    parser.add_argument('--print_diff', action='store_true')
    parser.add_argument('--legacy', action='store_true')
    parsed_args = parser.parse_args()
    if parsed_args.legacy:
        do_legacy_test(parsed_args.model_name[0], parsed_args.print_diff)
    else:
        do_test(parsed_args.model_name[0], parsed_args.bailout, parsed_args.print_diff)




