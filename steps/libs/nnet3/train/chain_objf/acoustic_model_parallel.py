

# Copyright 2016    Vijayaditya Peddinti.
#           2016    Vimal Manohar
# Apache 2.0.

""" This is a module with methods which will be used by scripts for training of
deep neural network acoustic model with chain objective.
"""

import logging
import math
import os, pdb, glob
import sys

import libs.common as common_lib
import libs.nnet3.train.common_parallel as common_train_lib

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def create_phone_lm(dir, tree_dir, run_opts, lm_opts=None):
    """Create a phone LM for chain training

    This method trains a phone LM for chain training using the alignments
    in "tree_dir"
    """
    try:
        f = open(tree_dir + "/num_jobs", 'r')
        num_ali_jobs = int(f.readline())
        assert num_ali_jobs > 0
    except:
        raise Exception("""There was an error getting the number of alignment
                        jobs from {0}/num_jobs""".format(tree_dir))

    alignments=' '.join(['{0}/ali.{1}.gz'.format(tree_dir, job)
                         for job in range(1, num_ali_jobs + 1)])

    common_lib.execute_command(
        """{command} {dir}/log/make_phone_lm.log \
            gunzip -c {alignments} \| \
            ali-to-phones {tree_dir}/final.mdl ark:- ark:- \| \
            chain-est-phone-lm {lm_opts} ark:- {dir}/phone_lm.fst""".format(
                command=run_opts.command, dir=dir,
                alignments=alignments,
                lm_opts=lm_opts if lm_opts is not None else '',
                tree_dir=tree_dir))


def create_denominator_fst(dir, tree_dir, run_opts):
    common_lib.execute_command(
        """copy-transition-model {tree_dir}/final.mdl \
                {dir}/0.trans_mdl""".format(dir=dir, tree_dir=tree_dir))
    common_lib.execute_command(
        """{command} {dir}/log/make_den_fst.log \
                   chain-make-den-fst {dir}/tree {dir}/0.trans_mdl \
                   {dir}/phone_lm.fst \
                   {dir}/den.fst {dir}/normalization.fst""".format(
                       dir=dir, command=run_opts.command))


def generate_chain_egs(dir, data, lat_dir, egs_dir,
                       left_context, right_context,
                       run_opts, stage=0,
                       left_tolerance=None, right_tolerance=None,
                       left_context_initial=-1, right_context_final=-1,
                       frame_subsampling_factor=3,
                       alignment_subsampling_factor=3,
                       online_ivector_dir=None,
                       frames_per_iter=20000, frames_per_eg_str="20", srand=0,
                       egs_opts=None, cmvn_opts=None, transform_dir=None):
    """Wrapper for steps/nnet3/chain/get_egs.sh

    See options in that script.
    """

    common_lib.execute_command(
        """steps/nnet3/chain/get_egs.sh {egs_opts} \
                --cmd "{command}" \
                --cmvn-opts "{cmvn_opts}" \
                --transform-dir "{transform_dir}" \
                --online-ivector-dir "{ivector_dir}" \
                --left-context {left_context} \
                --right-context {right_context} \
                --left-context-initial {left_context_initial} \
                --right-context-final {right_context_final} \
                --left-tolerance '{left_tolerance}' \
                --right-tolerance '{right_tolerance}' \
                --frame-subsampling-factor {frame_subsampling_factor} \
                --alignment-subsampling-factor {alignment_subsampling_factor} \
                --stage {stage} \
                --frames-per-iter {frames_per_iter} \
                --frames-per-eg {frames_per_eg_str} \
                --srand {srand} \
                {data} {dir} {lat_dir} {egs_dir}""".format(
                    command=run_opts.command,
                    cmvn_opts=cmvn_opts if cmvn_opts is not None else '',
                    transform_dir=(transform_dir
                                   if transform_dir is not None
                                   else ''),
                    ivector_dir=(online_ivector_dir
                                 if online_ivector_dir is not None
                                 else ''),
                    left_context=left_context,
                    right_context=right_context,
                    left_context_initial=left_context_initial,
                    right_context_final=right_context_final,
                    left_tolerance=(left_tolerance
                                    if left_tolerance is not None
                                    else ''),
                    right_tolerance=(right_tolerance
                                     if right_tolerance is not None
                                     else ''),
                    frame_subsampling_factor=frame_subsampling_factor,
                    alignment_subsampling_factor=alignment_subsampling_factor,
                    stage=stage, frames_per_iter=frames_per_iter,
                    frames_per_eg_str=frames_per_eg_str, srand=srand,
                    data=data, lat_dir=lat_dir, dir=dir, egs_dir=egs_dir,
                    egs_opts=egs_opts if egs_opts is not None else ''))


def train_new_models_parallel(dir, iter, srand, num_jobs,
                     num_archives_processed, num_archives,
                     raw_model_string, egs_dir,
                     apply_deriv_weights,
                     min_deriv_time, max_deriv_time_relative,
                     l2_regularize, xent_regularize, leaky_hmm_coefficient,
                     momentum, max_param_change,
                     shuffle_buffer_size, num_chunk_per_minibatch_str,
                     frame_subsampling_factor, run_opts, suffix,
                     backstitch_training_scale=0.0, backstitch_training_interval=1):
    """
    Called from train_one_iteration(), this method trains new models
    with 'num_jobs' jobs, and
    writes files like exp/tdnn_a/24.{1,2,3,..<num_jobs>}.raw

    We cannot easily use a single parallel SGE job to do the main training,
    because the computation of which archive and which --frame option
    to use for each job is a little complex, so we spawn each one separately.
    this is no longer true for RNNs as we use do not use the --frame option
    but we use the same script for consistency with FF-DNN code
    """

    deriv_time_opts = []
    if min_deriv_time is not None:
        deriv_time_opts.append("--optimization.min-deriv-time={0}".format(
                                    min_deriv_time))
    if max_deriv_time_relative is not None:
        deriv_time_opts.append("--optimization.max-deriv-time-relative={0}".format(
                                    int(max_deriv_time_relative)))

    threads = []
    # the GPU timing info is only printed if we use the --verbose=1 flag; this
    # slows down the computation slightly, so don't accumulate it on every
    # iteration.  Don't do it on iteration 0 either, because we use a smaller
    # than normal minibatch size, and people may get confused thinking it's
    # slower for iteration 0 because of the verbose option.
    verbose_opt = ("--verbose=1" if iter % 20 == 0 and iter > 0 else "")

    for job in range(1, num_jobs+1):
        # k is a zero-based index that we will derive the other indexes from.
        k = num_archives_processed + job - 1
        # work out the 1-based archive index.
        archive_index = (k % num_archives) + 1
        # previous : frame_shift = (k/num_archives) % frame_subsampling_factor
        frame_shift = ((archive_index + k/num_archives)
                       % frame_subsampling_factor)

        cache_io_opts = (("--read-cache={dir}/cache_{suffix}.{iter}".format(dir=dir,suffix=suffix,iter=iter)
                          if iter > 0 else "") +
                         (" --write-cache={0}/cache_{1}.{2}".format(dir,suffix,iter + 1)
                          if job == 1 else ""))

        thread = common_lib.background_command(
            """{command} {train_queue_opt} {dir}/log/train.{suffix}.{iter}.{job}.log \
                    nnet3-chain-train {parallel_train_opts} {verbose_opt} \
                    --apply-deriv-weights={app_deriv_wts} \
                    --l2-regularize={l2} --leaky-hmm-coefficient={leaky} \
                    {cache_io_opts}  --xent-regularize={xent_reg} \
                    {deriv_time_opts} \
                    --print-interval=10 --momentum={momentum} \
                    --max-param-change={max_param_change} \
                    --backstitch-training-scale={backstitch_training_scale} \
                    --backstitch-training-interval={backstitch_training_interval} \
                    --l2-regularize-factor={l2_regularize_factor} \
                    --srand={srand} \
                    "{raw_model}" {dir}/den.fst \
                    "ark,bg:nnet3-chain-copy-egs \
                        --frame-shift={fr_shft} \
                        ark:{egs_dir}/cegs.{archive_index}.ark ark:- | \
                        nnet3-chain-shuffle-egs --buffer-size={buf_size} \
                        --srand={srand} ark:- ark:- | nnet3-chain-merge-egs \
                        --minibatch-size={num_chunk_per_mb} ark:- ark:- |" \
                    {dir}/{next_iter}.{suffix}.{job}.raw""".format(
                        command=run_opts.command,
                        train_queue_opt=run_opts.train_queue_opt,
                        dir=dir, iter=iter, srand=iter + srand,
                        next_iter=iter + 1, job=job,
                        deriv_time_opts=" ".join(deriv_time_opts),
                        app_deriv_wts=apply_deriv_weights,
                        fr_shft=frame_shift, l2=l2_regularize,
                        xent_reg=xent_regularize, leaky=leaky_hmm_coefficient,
                        cache_io_opts=cache_io_opts,
                        parallel_train_opts=run_opts.parallel_train_opts,
                        verbose_opt=verbose_opt,
                        momentum=momentum, max_param_change=max_param_change,
                        backstitch_training_scale=backstitch_training_scale,
                        backstitch_training_interval=backstitch_training_interval,
                        l2_regularize_factor=1.0/num_jobs,
                        raw_model=raw_model_string, suffix=suffix,
                        egs_dir=egs_dir, archive_index=archive_index,
                        buf_size=shuffle_buffer_size,
                        num_chunk_per_mb=num_chunk_per_minibatch_str),
            require_zero_status=True)

        threads.append(thread)


    for thread in threads:
        thread.join()

def average_models(dir, iter, weights, run_opts):
    """ Called from steps/nnet3/chain/train.py for one iteration for
    neural network training with LF-MMI objective

    """
    nnets_list = []
    weights_list = []
    next_iter = iter+1
    
    for model_file in glob.glob(dir+'/'+str(next_iter)+'.*.raw'):
        print model_file
        if 'sad-sd-sl-res' in model_file:
            weights_list.append(weights.split(':')[0])
        elif 'cgn' in model_file:
            weights_list.append(weights.split(':')[1])
        elif 'vla' in model_file:
            weights_list.append(weights.split(':')[2])

        if os.path.exists(model_file):
            # we used to copy them with nnet3-am-copy --raw=true, but now
            # the raw-model-reading code discards the other stuff itself.
            nnets_list.append(model_file)
        else:
            print("{0}: warning: model file {1} does not exist "
                  "(final combination)".format(sys.argv[0], model_file))
    final_weights = ":".join(weights_list)
    common_train_lib.get_average_nnet_model_weighted(
        dir=dir, iter=iter,
        nnets_list=" ".join(nnets_list), weights=final_weights,
        run_opts=run_opts)

    try:
        os.system("rm -f {0}/{1}.*.raw".format(dir, iter + 1))
    except OSError:
        raise Exception("Error while trying to delete the raw models")

    new_model = "{0}/{1}.mdl".format(dir, iter + 1)

    if not os.path.isfile(new_model):
        raise Exception("Could not find {0}, at the end of "
                        "iteration {1}".format(new_model, iter))
    elif os.stat(new_model).st_size == 0:
        raise Exception("{0} has size 0. Something went wrong in "
                        "iteration {1}".format(new_model, iter))
    if os.path.exists("{0}/cache_*.{1}".format(dir, iter)):
        os.system("rm -f {0}/cache_*.{1}".format(dir, iter))

def train_one_iteration_parallel(dir, iter, srand, egs_dir,
                        num_jobs, num_archives_processed, num_archives,
                        learning_rate, shrinkage_value,
                        num_chunk_per_minibatch_str,
                        apply_deriv_weights, min_deriv_time,
                        max_deriv_time_relative,
                        l2_regularize, xent_regularize,
                        leaky_hmm_coefficient,
                        momentum, max_param_change, shuffle_buffer_size,
                        frame_subsampling_factor,
                        run_opts, suffix, dropout_edit_string="",
                        backstitch_training_scale=0.0, backstitch_training_interval=1):
    """ Called from steps/nnet3/chain/train.py for one iteration for
    neural network training with LF-MMI objective

    """

    # Set off jobs doing some diagnostics, in the background.
    # Use the egs dir from the previous iteration for the diagnostics
    # check if different iterations use the same random seed
    if os.path.exists('{0}/srand'.format(dir)):
        try:
            saved_srand = int(open('{0}/srand'.format(dir)).readline().strip())
        except (IOError, ValueError):
            logger.error("Exception while reading the random seed "
                         "for training")
            raise
        if srand != saved_srand:
            logger.warning("The random seed provided to this iteration "
                           "(srand={0}) is different from the one saved last "
                           "time (srand={1}). Using srand={0}.".format(
                               srand, saved_srand))
    else:
        with open('{0}/srand'.format(dir), 'w') as f:
            f.write(str(srand))

    # Sets off some background jobs to compute train and
    # validation set objectives
    compute_train_cv_probabilities(
        dir=dir, iter=iter, egs_dir=egs_dir, suffix=suffix,
        l2_regularize=l2_regularize, xent_regularize=xent_regularize,
        leaky_hmm_coefficient=leaky_hmm_coefficient, run_opts=run_opts)

    do_average = (iter > 0)

    raw_model_string = ("nnet3-am-copy --raw=true --learning-rate={0} "
                        "--scale={1} {2}/{3}.mdl - |".format(
                            learning_rate, shrinkage_value, dir, iter))

    if do_average:
        cur_num_chunk_per_minibatch_str = num_chunk_per_minibatch_str
        cur_max_param_change = max_param_change
    else:
        # on iteration zero, use a smaller minibatch size (and we will later
        # choose the output of just one of the jobs): the model-averaging isn't
        # always helpful when the model is changing too fast (i.e. it can worsen
        # the objective function), and the smaller minibatch size will help to
        # keep the update stable.
        cur_num_chunk_per_minibatch_str = common_train_lib.halve_minibatch_size_str(
            num_chunk_per_minibatch_str)
        cur_max_param_change = float(max_param_change) / math.sqrt(2)

    raw_model_string = raw_model_string + dropout_edit_string
    train_new_models_parallel(dir=dir, iter=iter, srand=srand, num_jobs=num_jobs,
                     num_archives_processed=num_archives_processed,
                     num_archives=num_archives,
                     raw_model_string=raw_model_string,
                     egs_dir=egs_dir,
                     apply_deriv_weights=apply_deriv_weights,
                     min_deriv_time=min_deriv_time,
                     max_deriv_time_relative=max_deriv_time_relative,
                     l2_regularize=l2_regularize,
                     xent_regularize=xent_regularize,
                     leaky_hmm_coefficient=leaky_hmm_coefficient,
                     momentum=momentum,
                     max_param_change=cur_max_param_change,
                     shuffle_buffer_size=shuffle_buffer_size,
                     num_chunk_per_minibatch_str=cur_num_chunk_per_minibatch_str,
                     frame_subsampling_factor=frame_subsampling_factor,
                     run_opts=run_opts,
                     suffix=suffix,
                     # linearly increase backstitch_training_scale during the
                     # first few iterations (hard-coded as 15)
                     backstitch_training_scale=(backstitch_training_scale *
                         iter / 15 if iter < 15 else backstitch_training_scale),
                     backstitch_training_interval=backstitch_training_interval)

    [models_to_average, best_model] = common_train_lib.get_successful_models(
         num_jobs, '{0}/log/train.{1}.{2}.%.log'.format(dir, suffix, iter))
    nnets_list = []
    for n in models_to_average:
        nnets_list.append("{0}/{1}.{2}.{3}.raw".format(dir, iter + 1, suffix, n))
    if do_average:
        # average the output of the different jobs.
        common_train_lib.get_average_nnet_model_parallel(
            dir=dir, iter=iter,
            nnets_list=" ".join(nnets_list), suffix=suffix,
            run_opts=run_opts)

    else:
        # choose the best model from different jobs
        common_train_lib.get_best_nnet_model_parallel(
            dir=dir, iter=iter,
            best_model_index=best_model, suffix=suffix,
            run_opts=run_opts)

    try:
        for i in range(1, num_jobs + 1):
            os.remove("{0}/{1}.{2}.{3}.raw".format(dir, iter + 1, suffix, i))
    except OSError:
        raise Exception("Error while trying to delete the raw models")

    new_model = "{0}/{1}.{2}.raw".format(dir, iter + 1, suffix)

    if not os.path.isfile(new_model):
        raise Exception("Could not find {0}, at the end of "
                        "iteration {1}".format(new_model, iter))
    elif os.stat(new_model).st_size == 0:
        raise Exception("{0} has size 0. Something went wrong in "
                        "iteration {1}".format(new_model, iter))
    if os.path.exists("{0}/cache_{1}.{2}".format(dir, suffix, iter)):
        os.remove("{0}/cache_{1}.{2}".format(dir, suffix, iter))


def check_for_required_files(feat_dir, tree_dir, lat_dir):
    files = ['{0}/feats.scp'.format(feat_dir), '{0}/ali.1.gz'.format(tree_dir),
             '{0}/final.mdl'.format(tree_dir), '{0}/tree'.format(tree_dir),
             '{0}/lat.1.gz'.format(lat_dir), '{0}/final.mdl'.format(lat_dir),
             '{0}/num_jobs'.format(lat_dir), '{0}/splice_opts'.format(lat_dir)]
    for file in files:
        if not os.path.isfile(file):
            raise Exception('Expected {0} to exist.'.format(file))


def compute_preconditioning_matrix(dir, egs_dir, num_lda_jobs, run_opts,
                                   max_lda_jobs=None, rand_prune=4.0,
                                   lda_opts=None):
    """ Function to estimate and write LDA matrix from cegs

    This function is exactly similar to the version in module
    libs.nnet3.train.frame_level_objf.common except this uses cegs instead of
    egs files.
    """
    if max_lda_jobs is not None:
        if num_lda_jobs > max_lda_jobs:
            num_lda_jobs = max_lda_jobs

    # Write stats with the same format as stats for LDA.
    common_lib.execute_command(
        """{command} JOB=1:{num_lda_jobs} {dir}/log/get_lda_stats.JOB.log \
                nnet3-chain-acc-lda-stats --rand-prune={rand_prune} \
                {dir}/init.raw "ark:{egs_dir}/cegs.JOB.ark" \
                {dir}/JOB.lda_stats""".format(
                    command=run_opts.command,
                    num_lda_jobs=num_lda_jobs,
                    dir=dir,
                    egs_dir=egs_dir,
                    rand_prune=rand_prune))

    # the above command would have generated dir/{1..num_lda_jobs}.lda_stats
    lda_stat_files = map(lambda x: '{0}/{1}.lda_stats'.format(dir, x),
                         range(1, num_lda_jobs + 1))

    common_lib.execute_command(
        """{command} {dir}/log/sum_transform_stats.log \
                sum-lda-accs {dir}/lda_stats {lda_stat_files}""".format(
                    command=run_opts.command,
                    dir=dir, lda_stat_files=" ".join(lda_stat_files)))

    for file in lda_stat_files:
        try:
            os.remove(file)
        except OSError:
            raise Exception("There was error while trying to remove "
                            "lda stat files.")
    # this computes a fixed affine transform computed in the way we described
    # in Appendix C.6 of http://arxiv.org/pdf/1410.7455v6.pdf; it's a scaled
    # variant of an LDA transform but without dimensionality reduction.

    common_lib.execute_command(
        """{command} {dir}/log/get_transform.log \
                nnet-get-feature-transform {lda_opts} {dir}/lda.mat \
                {dir}/lda_stats""".format(
                    command=run_opts.command, dir=dir,
                    lda_opts=lda_opts if lda_opts is not None else ""))

    common_lib.force_symlink("../lda.mat", "{0}/configs/lda.mat".format(dir))


def prepare_initial_acoustic_model(dir, run_opts, srand=-1, input_model=None):
    """ This function adds the first layer; It will also prepare the acoustic
        model with the transition model.
        If 'input_model' is specified, no initial network preparation(adding
        the first layer) is done and this model is used as initial 'raw' model
        instead of '0.raw' model to prepare '0.mdl' as acoustic model by adding the
        transition model.
    """
    if input_model is None:
        common_train_lib.prepare_initial_network(dir, run_opts,
                                                 srand=srand)

    # The model-format for a 'chain' acoustic model is just the transition
    # model and then the raw nnet, so we can use 'cat' to create this, as
    # long as they have the same mode (binary or not binary).
    # We ensure that they have the same mode (even if someone changed the
    # script to make one or both of them text mode) by copying them both
    # before concatenating them.
    common_lib.execute_command(
        """{command} {dir}/log/init_mdl.log \
                nnet3-am-init {dir}/0.trans_mdl {raw_mdl} \
                {dir}/0.mdl""".format(command=run_opts.command, dir=dir,
                                      raw_mdl=(input_model if input_model is not None
                                      else '{0}/0.raw'.format(dir))))


def compute_train_cv_probabilities(dir, iter, egs_dir, suffix, l2_regularize,
                                   xent_regularize, leaky_hmm_coefficient,
                                   run_opts):
    model = '{0}/{1}.mdl'.format(dir, iter)

    common_lib.background_command(
        """{command} {dir}/log/compute_prob_valid.{iter}.{suffix}.log \
                nnet3-chain-compute-prob --l2-regularize={l2} \
                --leaky-hmm-coefficient={leaky} --xent-regularize={xent_reg} \
                "nnet3-am-copy --raw=true {model} - |" {dir}/den.fst \
                "ark,bg:nnet3-chain-copy-egs ark:{egs_dir}/valid_diagnostic.cegs \
                    ark:- | nnet3-chain-merge-egs --minibatch-size=1:64 ark:- ark:- |" \
        """.format(command=run_opts.command, dir=dir, iter=iter, model=model,
                   l2=l2_regularize, leaky=leaky_hmm_coefficient,
                   xent_reg=xent_regularize, suffix=suffix,
                   egs_dir=egs_dir))

    common_lib.background_command(
        """{command} {dir}/log/compute_prob_train.{iter}.{suffix}.log \
                nnet3-chain-compute-prob --l2-regularize={l2} \
                --leaky-hmm-coefficient={leaky} --xent-regularize={xent_reg} \
                "nnet3-am-copy --raw=true {model} - |" {dir}/den.fst \
                "ark,bg:nnet3-chain-copy-egs ark:{egs_dir}/train_diagnostic.cegs \
                    ark:- | nnet3-chain-merge-egs --minibatch-size=1:64 ark:- ark:- |" \
        """.format(command=run_opts.command, dir=dir, iter=iter, model=model,
                   l2=l2_regularize, leaky=leaky_hmm_coefficient,
                   xent_reg=xent_regularize, suffix=suffix,
                   egs_dir=egs_dir))


def compute_progress(dir, iter, run_opts):

    prev_model = '{0}/{1}.mdl'.format(dir, iter - 1)
    model = '{0}/{1}.mdl'.format(dir, iter)

    common_lib.background_command(
        """{command} {dir}/log/progress.{iter}.log \
                nnet3-am-info {model} '&&' \
                nnet3-show-progress --use-gpu=no \
                    "nnet3-am-copy --raw=true {prev_model} - |" \
                    "nnet3-am-copy --raw=true {model} - |"
        """.format(command=run_opts.command,
                   dir=dir,
                   iter=iter,
                   model=model,
                   prev_model=prev_model))

def combine_parallel_models(dir, iter, next_iter, num_chunk_per_minibatch_str,
                   egs_dir, leaky_hmm_coefficient, l2_regularize,
                   xent_regularize, run_opts,
                   sum_to_one_penalty=1.0e-04):
    """ Function to do model combination

    In the nnet3 setup, the logic
    for doing averaging of subsets of the models in the case where
    there are too many models to reliably esetimate interpolation
    factors (max_models_combine) is moved into the nnet3-combine.
    """
    raw_model_strings = []
    logger.info("Combining the models.")

    # TODO: if it turns out the sum-to-one-penalty code is not useful,
    # remove support for it.
    for model_file in glob.glob(dir+'/'+str(next_iter)+'.*.raw'):
        print model_file
        if os.path.exists(model_file):
            # we used to copy them with nnet3-am-copy --raw=true, but now
            # the raw-model-reading code discards the other stuff itself.
            raw_model_strings.append(model_file)
        else:
            print("{0}: warning: model file {1} does not exist "
                  "(final combination)".format(sys.argv[0], model_file))

    # We reverse the order of the raw model strings so that the freshest one
    # goes first.  This is important for systems that include batch
    # normalization-- it means that the freshest batch-norm stats are used.
    # Since the batch-norm stats are not technically parameters, they are not
    # combined in the combination code, they are just obtained from the first
    # model.
    raw_model_strings = list(raw_model_strings)
    
    common_lib.execute_command(
        """{command} {combine_queue_opt} {dir}/log/combine.log \
                nnet3-chain-combine --num-iters={opt_iters} \
                --l2-regularize={l2} --leaky-hmm-coefficient={leaky} \
                --separate-weights-per-component={separate_weights} \
                --enforce-sum-to-one={hard_enforce} \
                --sum-to-one-penalty={penalty} \
                --enforce-positive-weights=true \
                --verbose=3 {dir}/den.fst {raw_models} \
                "ark,bg:nnet3-chain-copy-egs ark:{egs_dir}/combine.cegs ark:- | \
                    nnet3-chain-merge-egs --minibatch-size={num_chunk_per_mb} \
                    ark:- ark:- |" - \| \
                nnet3-am-copy --set-raw-nnet=- {dir}/{iter}.mdl \
                {dir}/{next_iter}.mdl""".format(
                    command=run_opts.command,
                    combine_queue_opt=run_opts.combine_queue_opt,
                    opt_iters=(20 if sum_to_one_penalty <= 0 else 80),
                    separate_weights=(sum_to_one_penalty > 0),
                    l2=l2_regularize, leaky=leaky_hmm_coefficient,
                    dir=dir, raw_models=" ".join(raw_model_strings),
                    hard_enforce=(sum_to_one_penalty <= 0),
                    penalty=sum_to_one_penalty,
                    num_chunk_per_mb=num_chunk_per_minibatch_str,
                    iter=iter,
                    next_iter=next_iter,
                    egs_dir=egs_dir))

def combine_parallel_models_weighted(dir, iter, next_iter, num_chunk_per_minibatch_str,
                   egs_dir, leaky_hmm_coefficient, l2_regularize, weights,
                   xent_regularize, run_opts,
                   sum_to_one_penalty=0.0):
    """ Function to do model combination

    In the nnet3 setup, the logic
    for doing averaging of subsets of the models in the case where
    there are too many models to reliably esetimate interpolation
    factors (max_models_combine) is moved into the nnet3-combine.
    """
    raw_model_strings = []
    logger.info("Combining the models.")

    # TODO: if it turns out the sum-to-one-penalty code is not useful,
    # remove support for it.
    for model_file in glob.glob(dir+'/'+str(next_iter)+'.*.raw'):
        print model_file
        if os.path.exists(model_file):
            # we used to copy them with nnet3-am-copy --raw=true, but now
            # the raw-model-reading code discards the other stuff itself.
            raw_model_strings.append(model_file)
        else:
            print("{0}: warning: model file {1} does not exist "
                  "(final combination)".format(sys.argv[0], model_file))

    # We reverse the order of the raw model strings so that the freshest one
    # goes first.  This is important for systems that include batch
    # normalization-- it means that the freshest batch-norm stats are used.
    # Since the batch-norm stats are not technically parameters, they are not
    # combined in the combination code, they are just obtained from the first
    # model.
    raw_model_strings = list(raw_model_strings)

    

def combine_models(dir, num_iters, models_to_combine, num_chunk_per_minibatch_str,
                   egs_dir, leaky_hmm_coefficient, l2_regularize,
                   xent_regularize, run_opts,
                   sum_to_one_penalty=0.0):
    """ Function to do model combination

    In the nnet3 setup, the logic
    for doing averaging of subsets of the models in the case where
    there are too many models to reliably esetimate interpolation
    factors (max_models_combine) is moved into the nnet3-combine.
    """
    raw_model_strings = []
    logger.info("Combining {0} models.".format(models_to_combine))

    models_to_combine.add(num_iters)

    # TODO: if it turns out the sum-to-one-penalty code is not useful,
    # remove support for it.

    for iter in sorted(models_to_combine):
        model_file = '{0}/{1}.mdl'.format(dir, iter)
        if os.path.exists(model_file):
            # we used to copy them with nnet3-am-copy --raw=true, but now
            # the raw-model-reading code discards the other stuff itself.
            raw_model_strings.append(model_file)
        else:
            print("{0}: warning: model file {1} does not exist "
                  "(final combination)".format(sys.argv[0], model_file))

    # We reverse the order of the raw model strings so that the freshest one
    # goes first.  This is important for systems that include batch
    # normalization-- it means that the freshest batch-norm stats are used.
    # Since the batch-norm stats are not technically parameters, they are not
    # combined in the combination code, they are just obtained from the first
    # model.
    raw_model_strings = list(reversed(raw_model_strings))

    common_lib.execute_command(
        """{command} {combine_queue_opt} {dir}/log/combine.log \
                nnet3-chain-combine --num-iters={opt_iters} \
                --l2-regularize={l2} --leaky-hmm-coefficient={leaky} \
                --separate-weights-per-component={separate_weights} \
                --enforce-sum-to-one={hard_enforce} \
                --sum-to-one-penalty={penalty} \
                --enforce-positive-weights=true \
                --verbose=3 {dir}/den.fst {raw_models} \
                "ark,bg:nnet3-chain-copy-egs ark:{egs_dir}/combine.cegs ark:- | \
                    nnet3-chain-merge-egs --minibatch-size={num_chunk_per_mb} \
                    ark:- ark:- |" - \| \
                nnet3-am-copy --set-raw-nnet=- {dir}/{num_iters}.mdl \
                {dir}/final.mdl""".format(
                    command=run_opts.command,
                    combine_queue_opt=run_opts.combine_queue_opt,
                    opt_iters=(20 if sum_to_one_penalty <= 0 else 80),
                    separate_weights=(sum_to_one_penalty > 0),
                    l2=l2_regularize, leaky=leaky_hmm_coefficient,
                    dir=dir, raw_models=" ".join(raw_model_strings),
                    hard_enforce=(sum_to_one_penalty <= 0),
                    penalty=sum_to_one_penalty,
                    num_chunk_per_mb=num_chunk_per_minibatch_str,
                    num_iters=num_iters,
                    egs_dir=egs_dir))

    # Compute the probability of the final, combined model with
    # the same subset we used for the previous compute_probs, as the
    # different subsets will lead to different probs.
    compute_train_cv_probabilities(
        dir=dir, iter='final', egs_dir=egs_dir,
        l2_regularize=l2_regularize, xent_regularize=xent_regularize,
        leaky_hmm_coefficient=leaky_hmm_coefficient,
        run_opts=run_opts)
