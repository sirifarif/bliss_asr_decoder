#!/usr/bin/env python

# Copyright 2016    Vijayaditya Peddinti.
#           2016    Vimal Manohar
# Apache 2.0.

""" This script is based on steps/nnet3/chain/train.sh
"""

import argparse
import logging
import os
import pprint
import shutil
import sys
import traceback

sys.path.insert(0, 'steps')
import libs.nnet3.train.common as common_train_lib
import libs.common as common_lib
import libs.nnet3.train.chain_objf.acoustic_model as chain_lib
import libs.nnet3.report.log_parse as nnet3_log_parse


logger = logging.getLogger('libs')
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s [%(pathname)s:%(lineno)s - "
                              "%(funcName)s - %(levelname)s ] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.info('Starting chain model trainer (train.py)')


def get_args():
    """ Get args from stdin.

    We add compulsary arguments as named arguments for readability

    The common options are defined in the object
    libs.nnet3.train.common.CommonParser.parser.
    See steps/libs/nnet3/train/common.py
    """

    parser = argparse.ArgumentParser(
        description="""Trains RNN and DNN acoustic models using the 'chain'
        objective function.""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        conflict_handler='resolve',
        parents=[common_train_lib.CommonParser().parser])

    # egs extraction options
    parser.add_argument("--egs.chunk-width", type=str, dest='chunk_width',
                        default="20",
                        help="""Number of frames per chunk in the examples
                        used to train the RNN.   Caution: if you double this you
                        should halve --trainer.samples-per-iter.  May be
                        a comma-separated list of alternatives: first width
                        is the 'principal' chunk-width, used preferentially""")

    # chain options
    parser.add_argument("--chain.lm-opts", type=str, dest='lm_opts',
                        default=None, action=common_lib.NullstrToNoneAction,
                        help="options to be be passed to chain-est-phone-lm")
    parser.add_argument("--chain.l2-regularize", type=float,
                        dest='l2_regularize', default=0.0,
                        help="""Weight of regularization function which is the
                        l2-norm of the output of the network. It should be used
                        without the log-softmax layer for the outputs.  As
                        l2-norm of the log-softmax outputs can dominate the
                        objective function.""")
    parser.add_argument("--chain.xent-regularize", type=float,
                        dest='xent_regularize', default=0.0,
                        help="Weight of regularization function which is the "
                        "cross-entropy cost the outputs.")
    parser.add_argument("--chain.right-tolerance", type=int,
                        dest='right_tolerance', default=5, help="")
    parser.add_argument("--chain.left-tolerance", type=int,
                        dest='left_tolerance', default=5, help="")
    parser.add_argument("--chain.leaky-hmm-coefficient", type=float,
                        dest='leaky_hmm_coefficient', default=0.00001,
                        help="")
    parser.add_argument("--chain.apply-deriv-weights", type=str,
                        dest='apply_deriv_weights', default=True,
                        action=common_lib.StrToBoolAction,
                        choices=["true", "false"],
                        help="")
    parser.add_argument("--chain.frame-subsampling-factor", type=int,
                        dest='frame_subsampling_factor', default=3,
                        help="ratio of frames-per-second of features we "
                        "train on, to chain model's output")
    parser.add_argument("--chain.alignment-subsampling-factor", type=int,
                        dest='alignment_subsampling_factor',
                        default=3,
                        help="ratio of frames-per-second of input "
                        "alignments to chain model's output")
    parser.add_argument("--chain.left-deriv-truncate", type=int,
                        dest='left_deriv_truncate',
                        default=None,
                        help="Deprecated. Kept for back compatibility")

    # trainer options
    parser.add_argument("--trainer.input-model", type=str,
                        dest='input_model', default=None,
                        action=common_lib.NullstrToNoneAction,
                        help="If specified, this model is used as initial "
                             "'raw' model (0.raw in the script) instead of "
                             "initializing the model from the xconfig. "
                             "Also configs dir is not expected to exist "
                             "and left/right context is computed from this "
                             "model.")
    parser.add_argument("--trainer.num-epochs", type=float, dest='num_epochs',
                        default=10.0,
                        help="Number of epochs to train the model")
    parser.add_argument("--trainer.frames-per-iter", type=int,
                        dest='frames_per_iter', default=800000,
                        help="""Each iteration of training, see this many
                        [input] frames per job.  This option is passed to
                        get_egs.sh.  Aim for about a minute of training
                        time""")

    parser.add_argument("--trainer.num-chunk-per-minibatch", type=str,
                        dest='num_chunk_per_minibatch', default='128',
                        help="""Number of sequences to be processed in
                        parallel every minibatch.  May be a more general
                        rule as accepted by the --minibatch-size option of
                        nnet3-merge-egs; run that program without args to see
                        the format.""")

    # Parameters for the optimization
    parser.add_argument("--trainer.optimization.initial-effective-lrate",
                        type=float, dest='initial_effective_lrate',
                        default=0.0002,
                        help="Learning rate used during the initial iteration")
    parser.add_argument("--trainer.optimization.final-effective-lrate",
                        type=float, dest='final_effective_lrate',
                        default=0.00002,
                        help="Learning rate used during the final iteration")
    parser.add_argument("--trainer.optimization.shrink-value", type=float,
                        dest='shrink_value', default=1.0,
                        help="""Scaling factor used for scaling the parameter
                        matrices when the derivative averages are below the
                        shrink-threshold at the non-linearities.  E.g. 0.99.
                        Only applicable when the neural net contains sigmoid or
                        tanh units.""")
    parser.add_argument("--trainer.optimization.shrink-saturation-threshold",
                        type=float,
                        dest='shrink_saturation_threshold', default=0.40,
                        help="""Threshold that controls when we apply the
                        'shrinkage' (i.e. scaling by shrink-value).  If the
                        saturation of the sigmoid and tanh nonlinearities in
                        the neural net (as measured by
                        steps/nnet3/get_saturation.pl) exceeds this threshold
                        we scale the parameter matrices with the
                        shrink-value.""")
    # RNN-specific training options
    parser.add_argument("--trainer.deriv-truncate-margin", type=int,
                        dest='deriv_truncate_margin', default=None,
                        help="""(Relevant only for recurrent models). If
                        specified, gives the margin (in input frames) around
                        the 'required' part of each chunk that the derivatives
                        are backpropagated to. If unset, the derivatives are
                        backpropagated all the way to the boundaries of the
                        input data. E.g. 8 is a reasonable setting. Note: the
                        'required' part of the chunk is defined by the model's
                        {left,right}-context.""")

    # General options
    parser.add_argument("--feat-dir", type=str, required=True,
                        help="Directory with features used for training "
                        "the neural network.")
    parser.add_argument("--tree-dir", type=str, required=True,
                        help="""Directory containing the tree to use for this
                        model (we also expect final.mdl and ali.*.gz in that
                        directory""")
    parser.add_argument("--lat-dir", type=str, required=True,
                        help="Directory with numerator lattices "
                        "used for training the neural network.")
    parser.add_argument("--dir", type=str, required=True,
                        help="Directory to store the models and "
                        "all other files.")

    print(' '.join(sys.argv))
    print(sys.argv)

    args = parser.parse_args()

    [args, run_opts] = process_args(args)

    return [args, run_opts]


def process_args(args):
    """ Process the options got from get_args()
    """

    if not common_train_lib.validate_chunk_width(args.chunk_width):
        raise Exception("--egs.chunk-width has an invalid value")

    if not common_train_lib.validate_minibatch_size_str(args.num_chunk_per_minibatch):
        raise Exception("--trainer.num-chunk-per-minibatch has an invalid value")

    if args.chunk_left_context < 0:
        raise Exception("--egs.chunk-left-context should be non-negative")

    if args.chunk_right_context < 0:
        raise Exception("--egs.chunk-right-context should be non-negative")

    if args.left_deriv_truncate is not None:
        args.deriv_truncate_margin = -args.left_deriv_truncate
        logger.warning(
            "--chain.left-deriv-truncate (deprecated) is set by user, and "
            "--trainer.deriv-truncate-margin is set to negative of that "
            "value={0}. We recommend using the option "
            "--trainer.deriv-truncate-margin.".format(
                args.deriv_truncate_margin))

    if (not os.path.exists(args.dir)
            or (not os.path.exists(args.dir+"/configs") and
                not os.path.exists(args.input_model))):
        raise Exception("This script expects {0} to exist. Also either "
                        "--trainer.input-model option as initial 'raw' model "
                        "(used as 0.raw in the script) should be supplied or "
                        "{0}/configs directory which is the output of "
                        "make_configs.py script should be provided."
                        "".format(args.dir))

    if args.transform_dir is None:
        args.transform_dir = args.lat_dir
    # set the options corresponding to args.use_gpu
    run_opts = common_train_lib.RunOpts()
    if args.use_gpu:
        if not common_lib.check_if_cuda_compiled():
            logger.warning(
                """You are running with one thread but you have not compiled
                   for CUDA.  You may be running a setup optimized for GPUs.
                   If you have GPUs and have nvcc installed, go to src/ and do
                   ./configure; make""")

        run_opts.train_queue_opt = "--gpu 1"
        run_opts.parallel_train_opts = ""
        run_opts.combine_queue_opt = "--gpu 1"

    else:
        logger.warning("Without using a GPU this will be very slow. "
                       "nnet3 does not yet support multiple threads.")

        run_opts.train_queue_opt = ""
        run_opts.parallel_train_opts = "--use-gpu=no"
        run_opts.combine_queue_opt = ""

    run_opts.command = args.command
    run_opts.egs_command = (args.egs_command
                            if args.egs_command is not None else
                            args.command)

    return [args, run_opts]


def model_init(args, run_opts):
    """ The main function for training.

    Args:
        args: a Namespace object with the required parameters
            obtained from the function process_args()
        run_opts: RunOpts object obtained from the process_args()
    """

    arg_string = pprint.pformat(vars(args))
    logger.info("Arguments for the experiment\n{0}".format(arg_string))

    # Check files
    chain_lib.check_for_required_files(args.feat_dir, args.tree_dir,
                                       args.lat_dir)

    # Set some variables.
    num_jobs = common_lib.get_number_of_jobs(args.tree_dir)
    feat_dim = common_lib.get_feat_dim(args.feat_dir)
    ivector_dim = common_lib.get_ivector_dim(args.online_ivector_dir)
    ivector_id = common_lib.get_ivector_extractor_id(args.online_ivector_dir)

    # split the training data into parts for individual jobs
    # we will use the same number of jobs as that used for alignment
    common_lib.execute_command("utils/split_data.sh {0} {1}"
                               "".format(args.feat_dir, num_jobs))
    with open('{0}/num_jobs'.format(args.dir), 'w') as f:
        f.write(str(num_jobs))

    if args.input_model is None:
        config_dir = '{0}/configs'.format(args.dir)
        var_file = '{0}/vars'.format(config_dir)

        variables = common_train_lib.parse_generic_config_vars_file(var_file)
    else:
        # If args.input_model is specified, the model left and right contexts
        # are computed using input_model.
        variables = common_train_lib.get_input_model_info(args.input_model)

    # Set some variables.
    try:
        model_left_context = variables['model_left_context']
        model_right_context = variables['model_right_context']
    except KeyError as e:
        raise Exception("KeyError {0}: Variables need to be defined in "
                        "{1}".format(str(e), '{0}/configs'.format(args.dir)))

    left_context = args.chunk_left_context + model_left_context
    right_context = args.chunk_right_context + model_right_context
    left_context_initial = (args.chunk_left_context_initial + model_left_context if
                            args.chunk_left_context_initial >= 0 else -1)
    right_context_final = (args.chunk_right_context_final + model_right_context if
                           args.chunk_right_context_final >= 0 else -1)


    egs_left_context = left_context + args.frame_subsampling_factor / 2
    egs_right_context = right_context + args.frame_subsampling_factor / 2
    # note: the '+ args.frame_subsampling_factor / 2' is to allow for the
    # fact that we'll be shifting the data slightly during training to give
    # variety to the training data.
    egs_left_context_initial = (left_context_initial +
                                args.frame_subsampling_factor / 2 if
                                left_context_initial >= 0 else -1)
    egs_right_context_final = (right_context_final +
                               args.frame_subsampling_factor / 2 if
                               right_context_final >= 0 else -1)

    default_egs_dir = '{0}/egs'.format(args.dir)
    num_jobs = common_lib.get_number_of_jobs(args.tree_dir)
    feat_dim = common_lib.get_feat_dim(args.feat_dir)
    ivector_dim = common_lib.get_ivector_dim(args.online_ivector_dir)
    ivector_id = common_lib.get_ivector_extractor_id(args.online_ivector_dir)

    # split the training data into parts for individual jobs
    # we will use the same number of jobs as that used for alignment
    common_lib.execute_command("utils/split_data.sh {0} {1}"
                               "".format(args.feat_dir, num_jobs))
    with open('{0}/num_jobs'.format(args.dir), 'w') as f:
        f.write(str(num_jobs))

    if args.input_model is None:
        config_dir = '{0}/configs'.format(args.dir)
        var_file = '{0}/vars'.format(config_dir)

        variables = common_train_lib.parse_generic_config_vars_file(var_file)
    else:
        # If args.input_model is specified, the model left and right contexts
        # are computed using input_model.
        variables = common_train_lib.get_input_model_info(args.input_model)

    # Set some variables.
    try:
        model_left_context = variables['model_left_context']
        model_right_context = variables['model_right_context']
    except KeyError as e:
        raise Exception("KeyError {0}: Variables need to be defined in "
                        "{1}".format(str(e), '{0}/configs'.format(args.dir)))

    left_context = args.chunk_left_context + model_left_context
    right_context = args.chunk_right_context + model_right_context
    left_context_initial = (args.chunk_left_context_initial + model_left_context if
                            args.chunk_left_context_initial >= 0 else -1)
    right_context_final = (args.chunk_right_context_final + model_right_context if
                           args.chunk_right_context_final >= 0 else -1)

    if args.egs_dir is None:
        egs_dir = default_egs_dir
    else:
        egs_dir = args.egs_dir

    [egs_left_context, egs_right_context,
     frames_per_eg_str, num_archives] = (
         common_train_lib.verify_egs_dir(egs_dir, feat_dim,
                                         ivector_dim, ivector_id,
                                         egs_left_context, egs_right_context,
                                         egs_left_context_initial,
                                         egs_right_context_final))
    assert(args.chunk_width == frames_per_eg_str)
    num_archives_expanded = num_archives * args.frame_subsampling_factor

    if (args.num_jobs_final > num_archives_expanded):
        raise Exception('num_jobs_final cannot exceed the '
                        'expanded number of archives')

    # copy the properties of the egs to dir for
    # use during decoding
    logger.info("Copying the properties from {0} to {1}".format(egs_dir, args.dir))
    common_train_lib.copy_egs_properties_to_exp_dir(egs_dir, args.dir)

    if ((args.stage <= -2) and (os.path.exists(args.dir+"/configs/init.config"))
            and (args.input_model is None)):
        logger.info('Computing the preconditioning matrix for input features')

        chain_lib.compute_preconditioning_matrix(
            args.dir, egs_dir, num_archives, run_opts,
            max_lda_jobs=args.max_lda_jobs,
            rand_prune=args.rand_prune)

    if (args.stage <= -1):
        logger.info("Preparing the initial acoustic model.")
        chain_lib.prepare_initial_acoustic_model(args.dir, run_opts,
                                                 input_model=args.input_model)

    with open("{0}/frame_subsampling_factor".format(args.dir), "w") as f:
        f.write(str(args.frame_subsampling_factor))

    # set num_iters so that as close as possible, we process the data
    # $num_epochs times, i.e. $num_iters*$avg_num_jobs) ==
    # $num_epochs*$num_archives, where
    # avg_num_jobs=(num_jobs_initial+num_jobs_final)/2.
    num_archives_to_process = int(args.num_epochs * num_archives_expanded)
    num_archives_processed = 0
    num_iters = ((num_archives_to_process * 2)
                 / (args.num_jobs_initial + args.num_jobs_final))

    # If do_final_combination is True, compute the set of models_to_combine.
    # Otherwise, models_to_combine will be none.
    if args.do_final_combination:
        models_to_combine = common_train_lib.get_model_combine_iters(
            num_iters, args.num_epochs,
            num_archives_expanded, args.max_models_combine,
            args.num_jobs_final)
    else:
        models_to_combine = None

    min_deriv_time = None
    max_deriv_time_relative = None
    if args.deriv_truncate_margin is not None:
        min_deriv_time = -args.deriv_truncate_margin - model_left_context
        max_deriv_time_relative = \
           args.deriv_truncate_margin + model_right_context

    logger.info("Training will run for {0} epochs = "
                "{1} iterations".format(args.num_epochs, num_iters))

def main():
    [args, run_opts] = get_args()
    try:
        model_init(args, run_opts)
        common_lib.wait_for_background_commands()
    except BaseException as e:
        # look for BaseException so we catch KeyboardInterrupt, which is
        # what we get when a background thread dies.
        if args.email is not None:
            message = ("Training session for experiment {dir} "
                       "died due to an error.".format(dir=args.dir))
            common_lib.send_mail(message, message, args.email)
        if not isinstance(e, KeyboardInterrupt):
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
