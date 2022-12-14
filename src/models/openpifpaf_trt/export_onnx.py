"""Export a checkpoint as an ONNX model.

Applies onnx utilities to improve the exported model and
also tries to simplify the model with onnx-simplifier.

https://github.com/onnx/onnx/blob/master/docs/PythonAPIOverview.md
https://github.com/daquexian/onnx-simplifier
"""

import argparse
import logging
import shutil

import torch

import openpifpaf.network

#import network

try:
    import onnx
    import onnx.utils
except ImportError:
    onnx = None

try:
    import onnxsim
except ImportError:
    onnxsim = None

LOG = logging.getLogger(__name__)


def image_size_warning(basenet_stride, input_w, input_h):
    if input_w % basenet_stride != 1:
        LOG.warning(
            'input width (%d) should be a multiple of basenet '
            'stride (%d) + 1: closest are %d and %d',
            input_w, basenet_stride,
            (input_w - 1) // basenet_stride * basenet_stride + 1,
            ((input_w - 1) // basenet_stride + 1) * basenet_stride + 1,
        )

    if input_h % basenet_stride != 1:
        LOG.warning(
            'input height (%d) should be a multiple of basenet '
            'stride (%d) + 1: closest are %d and %d',
            input_h, basenet_stride,
            (input_h - 1) // basenet_stride * basenet_stride + 1,
            ((input_h - 1) // basenet_stride + 1) * basenet_stride + 1,
        )


def apply(model, outfile, verbose=True, input_w=129, input_h=97):
    image_size_warning(model.base_net.stride, input_w, input_h)

    # configure
    openpifpaf.network.heads.CompositeField3.inplace_ops = False

    dummy_input = torch.randn(1, 3, input_h, input_w)

    torch.onnx.export(
        model, dummy_input, outfile, verbose=verbose,
        input_names=['input_batch'], output_names=['cif', 'caf'],
        # keep_initializers_as_inputs=True,
        # opset_version=10,
        do_constant_folding=True,
        export_params=True,
        # dynamic_axes={  # TODO: gives warnings
        #     'input_batch': {0: 'batch', 2: 'height', 3: 'width'},
        #     'pif_c': {0: 'batch', 2: 'fheight', 3: 'fwidth'},
        #     'pif_r': {0: 'batch', 3: 'fheight', 4: 'fwidth'},
        #     'pif_b': {0: 'batch', 2: 'fheight', 3: 'fwidth'},
        #     'pif_s': {0: 'batch', 2: 'fheight', 3: 'fwidth'},

        #     'paf_c': {0: 'batch', 2: 'fheight', 3: 'fwidth'},
        #     'paf_r1': {0: 'batch', 3: 'fheight', 4: 'fwidth'},
        #     'paf_b1': {0: 'batch', 2: 'fheight', 3: 'fwidth'},
        #     'paf_r2': {0: 'batch', 3: 'fheight', 4: 'fwidth'},
        #     'paf_b2': {0: 'batch', 2: 'fheight', 3: 'fwidth'},
        # },
    )


def optimize(infile, outfile=None):
    if outfile is None:
        assert infile.endswith('.onnx')
        outfile = infile
        infile = infile.replace('.onnx', '.unoptimized.onnx')
        shutil.copyfile(outfile, infile)

    model = onnx.load(infile)
    optimized_model = onnx.optimizer.optimize(model)
    onnx.save(optimized_model, outfile)


def check(modelfile):
    model = onnx.load(modelfile)
    onnx.checker.check_model(model)


def polish(infile, outfile=None):
    if outfile is None:
        assert infile.endswith('.onnx')
        outfile = infile
        infile = infile.replace('.onnx', '.unpolished.onnx')
        shutil.copyfile(outfile, infile)

    model = onnx.load(infile)
    polished_model = onnx.utils.polish_model(model)
    onnx.save(polished_model, outfile)


def simplify(infile, outfile=None):
    if outfile is None:
        assert infile.endswith('.onnx')
        outfile = infile
        infile = infile.replace('.onnx', '.unsimplified.onnx')
        shutil.copyfile(outfile, infile)

    simplified_model, check_ok = onnxsim.simplify(infile, check_n=3, perform_optimization=False)
    assert check_ok
    onnx.save(simplified_model, outfile)


class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter,
                      argparse.RawDescriptionHelpFormatter):
    pass


def main():
    parser = argparse.ArgumentParser(
        prog='python3 -m openpifpaf.export_onnx',
        description=__doc__,
        formatter_class=CustomFormatter,
    )
    parser.add_argument('--version', action='version',
                        version='OpenPifPaf {version}'.format(version='1'))

    parser.add_argument('--checkpoint', default='resnet50')
    parser.add_argument('--outfile', default='openpifpaf-resnet50.onnx')
    parser.add_argument('--simplify', dest='simplify', default=False, action='store_true')
    parser.add_argument('--polish', dest='polish', default=False, action='store_true',
                        help='runs checker, optimizer and shape inference')
    parser.add_argument('--optimize', dest='optimize', default=False, action='store_true')
    parser.add_argument('--check', dest='check', default=False, action='store_true')
    parser.add_argument('--input-width', type=int, default=129)
    parser.add_argument('--input-height', type=int, default=97)
    args = parser.parse_args()

    model, _ = openpifpaf.network.factory(checkpoint=args.checkpoint)
    apply(model, args.outfile, input_w=args.input_width, input_h=args.input_height)
    if args.simplify:
        simplify(args.outfile)
    if args.optimize:
        optimize(args.outfile)
    if args.polish:
        polish(args.outfile)
    if args.check:
        check(args.outfile)


if __name__ == '__main__':
    main()
