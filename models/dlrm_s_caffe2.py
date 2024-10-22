# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Description: an implementation of a deep learning recommendation model (DLRM)
# The model input consists of dense and sparse features. The former is a vector
# of floating point values. The latter is a list of sparse indices into
# embedding tables, which consist of vectors of floating point values.
# The selected vectors are passed to mlp networks denoted by triangles,
# in some cases the vectors are interacted through operators (Ops).
#
# output:
#                         vector of values
# model:                        |
#                              /\
#                             /__\
#                               |
#       _____________________> Op  <___________________
#     /                         |                      \
#    /\                        /\                      /\
#   /__\                      /__\           ...      /__\
#    |                          |                       |
#    |                         Op                      Op
#    |                    ____/__\_____           ____/__\____
#    |                   |_Emb_|____|__|    ...  |_Emb_|__|___|
# input:
# [ dense features ]     [sparse indices] , ..., [sparse indices]
#
# More precise definition of model layers:
# 1) fully connected layers of an mlp
# z = f(y)
# y = Wx + b
#
# 2) embedding lookup (for a list of sparse indices p=[p1,...,pk])
# z = Op(e1,...,ek)
# obtain vectors e1=E[:,p1], ..., ek=E[:,pk]
#
# 3) Operator Op can be one of the following
# Sum(e1,...,ek) = e1 + ... + ek
# Dot(e1,...,ek) = [e1'e1, ..., e1'ek, ..., ek'e1, ..., ek'ek]
# Cat(e1,...,ek) = [e1', ..., ek']'
# where ' denotes transpose operation
#
# References:
# [1] Maxim Naumov, Dheevatsa Mudigere, Hao-Jun Michael Shi, Jianyu Huang,
# Narayanan Sundaram, Jongsoo Park, Xiaodong Wang, Udit Gupta, Carole-Jean Wu,
# Alisson G. Azzolini, Dmytro Dzhulgakov, Andrey Mallevich, Ilia Cherniavskii,
# Yinghai Lu, Raghuraman Krishnamoorthi, Ansha Yu, Volodymyr Kondratenko,
# Stephanie Pereia, Xianjie Chen, Wenlin Chen, Vijay Rao, Bill Jia, Liang Xiong,
# Misha Smelyanskiy, "Deep Learning Recommendation Model for Personalization and
# Recommendation Systems", CoRR, arXiv:1906.00091, 2019

from __future__ import absolute_import, division, print_function, unicode_literals

import functools

# others
import operator
import time
from ctypes import *
import Queue
from threading import Thread
from lru import LRU
import math

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors

# numpy
import numpy as np

# cProfile
import cProfile

# caffe2
from caffe2.proto import caffe2_pb2
from caffe2.python import brew, core, dyndep, model_helper, net_drawer, workspace
from numpy import random as ra
import caffe2.python._import_c_extension as C
import sys

# pickle
import pickle


dyndep.InitOpsLibrary( "//caffe2/caffe2/fb/operators/intra_op_parallel:intra_op_parallel_ops" )
# =============================================================================
# Define wrapper for dlrm in Caffe2
# This is to decouple input queues for DLRM network and the DLRM network itself
# =============================================================================
class DLRM_Wrapper(object):
    def FeedBlobWrapper(self, tag, val):
        if self.gpu_en:
            _d = core.DeviceOption(caffe2_pb2.CUDA, 0)
            with core.DeviceScope(_d):
                workspace.FeedBlob(tag, val, device_option=_d)
            print("(Queue) Feeding into GPU")
        else:
            workspace.FeedBlob(tag, val)
            print("(Queue) Feeding into CPU")

    def __init__(
        self,
        cli_args,
        model=None,
        tag=None,
        enable_prof=False,
    ):
        super(DLRM_Wrapper, self).__init__()
        self.args = cli_args

        # GPU Enable Flags
        gpu_en = self.args.use_gpu

        if gpu_en:
            device_opt = core.DeviceOption(caffe2_pb2.CUDA, 0)
            ngpus = C.num_cuda_devices  # 1
            print("(Wrapper) Using {} GPU(s)...".format(ngpus))
        else:
            device_opt = core.DeviceOption(caffe2_pb2.CPU)
            print("(Wrapper) Using CPU...")

        self.gpu_en = gpu_en

        num_tables = len(cli_args.arch_embedding_size.split("-"))

        # We require 3 datastructures in caffe2 to enable non-blocking inputs for DLRM
        # At a high-level each input needs an input queue. Inputs are enqueued
        # when they arrive on the "server" or "core" and dequeued by the
        # model's inference engine
        # Input Blob -> Input Net -> ID Q ===> DLRM model
        self.id_qs          = []
        self.id_input_blobs = []
        self.id_input_nets  = []

        # Same thing for the lengths inputs
        self.len_qs          = []
        self.len_input_blobs = []
        self.len_input_nets  = []

        for i in range(num_tables):

            q, input_blob, net = self.build_dlrm_sparse_queue(tag="id", qid=i)
            self.id_qs.append(q)
            self.id_input_blobs.append(input_blob)
            self.id_input_nets.append(net)

            q, input_blob, net = self.build_dlrm_sparse_queue(tag="len", qid=i)
            self.len_qs.append(q)
            self.len_input_blobs.append(input_blob)
            self.len_input_nets.append(net)

        self.fc_q, self.fc_input_blob, self.fc_input_net = self.build_dlrm_fc_queue()

        if self.args.queue:
            with core.DeviceScope(device_opt):
                self.dlrm = DLRM_Net(cli_args, model, tag, enable_prof,
                                     id_qs = self.id_qs,
                                     len_qs = self.len_qs,
                                     fc_q   = self.fc_q)
        else:
            with core.DeviceScope(device_opt):
                self.dlrm = DLRM_Net(cli_args, model, tag, enable_prof)


    def create(self, X, S_lengths, S_indices, T):
        if self.args.queue:
            self.dlrm.create(X, S_lengths, S_indices, T,
                             id_qs = self.id_qs,
                             len_qs = self.len_qs)
        else:
            self.dlrm.create(X, S_lengths, S_indices, T)


    # Run the Queues to provide inputs to DLRM model
    def run_queues(self, ids, lengths, fc, batch_size):
        # Dense features
        self.FeedBlobWrapper(self.fc_input_blob, fc)
        workspace.RunNetOnce(self.fc_input_net.Proto())

        # Sparse features
        num_tables = len(self.args.arch_embedding_size.split("-"))
        for i in range(num_tables):
           self.FeedBlobWrapper( self.id_input_blobs[i], ids[i])
           workspace.RunNetOnce( self.id_input_nets[i].Proto() )

           self.FeedBlobWrapper( self.len_input_blobs[i], lengths[i])
           Workspace.RunNetOnce( self.len_input_nets[i].Proto() )
    # =========================================================================
    # Helper functions to build queues for DLRM inputs (IDs, Lengths, FC)
    # in order to decouple blocking input operations
    # =========================================================================
    def build_dlrm_sparse_queue(self, tag = "id", qid = None):
        q_net_name = tag + '_q_init_' + str(qid)
        q_net = core.Net(q_net_name)

        q_input_blob_name = tag + '_q_blob_' + str(qid)

        with core.DeviceScope(core.DeviceOption(caffe2_pb2.CPU)):
            q = q_net.CreateBlobsQueue([],q_input_blob_name,num_blobs=1,capacity=8)

        workspace.RunNetOnce(q_net)

        input_blob_name = tag + '_inputs_' + str(qid)
        input_net = core.Net(tag + '_input_net_' + str(qid))
        input_net.EnqueueBlobs([q, input_blob_name], [input_blob_name])

        return q, input_blob_name, input_net

    def build_dlrm_fc_queue(self, ):
        fc_q_net_name = 'fc_q_init'
        fc_q_net = core.Net(fc_q_net_name)

        fc_q_input_blob_name = 'fc_q_blob'

        with core.DeviceScope(core.DeviceOption(caffe2_pb2.CPU)):
            fc_q = fc_q_net.CreateBlobsQueue([],fc_q_input_blob_name,num_blobs=1,capacity=8)

        workspace.RunNetOnce(fc_q_net)

        fc_input_blob_name = 'fc_inputs'
        fc_input_net = core.Net('fc_input_net')
        fc_input_net.EnqueueBlobs([fc_q, fc_input_blob_name], [fc_input_blob_name])

        return fc_q, fc_input_blob_name, fc_input_net


class DLRM_Net(object):
    def FeedBlobWrapper(self, tag, val):
        if self.gpu_en:
            _d = core.DeviceOption(caffe2_pb2.CUDA, 0)
            # with core.DeviceScope(_d):
            workspace.FeedBlob(tag, val, device_option=_d)
            # print("Feeding into GPU")
        else:
            workspace.FeedBlob(tag, val)
            # print("Feeding into CPU")

    def create_mlp(self, ln, sigmoid_layer, model, tag, fc_q = None):
        (tag_layer, tag_in, tag_out) = tag

        # build MLP layer by layer
        layers = []
        weights = []
        for i in range(1, ln.size):
            n = ln[i - 1]
            m = ln[i]

            # create tags
            tag_fc_w = tag_layer + ":::" + "fc" + str(i) + "_w"
            tag_fc_b = tag_layer + ":::" + "fc" + str(i) + "_b"
            tag_fc_y = tag_layer + ":::" + "fc" + str(i) + "_y"
            tag_fc_z = tag_layer + ":::" + "fc" + str(i) + "_z"
            if i == ln.size - 1:
                tag_fc_z = tag_out
            weights.append(tag_fc_w)
            weights.append(tag_fc_b)

            # initialize the weights
            # approach 1: custom Xavier input, output or two-sided fill
            mean = 0.0 # std_dev = np.sqrt(variance)
            std_dev = np.sqrt(2 / (m + n)) # np.sqrt(1 / m) # np.sqrt(1 / n)
            W = np.random.normal(mean, std_dev, size=(m, n)).astype(np.float32)
            std_dev = np.sqrt(1 / m) # np.sqrt(2 / (m + 1))
            b = np.random.normal(mean, std_dev, size=m).astype(np.float32)
            # with core.DeviceScope(core.DeviceOption(caffe2_pb2.CUDA, 0)):
            self.FeedBlobWrapper(tag_fc_w, W)
            self.FeedBlobWrapper(tag_fc_b, b)

            # approach 1: construct fully connected operator using model.net
            if self.args.queue and (fc_q is not None) and (i == 1):
                # Dequeue lengths vector as well
                model.net.DequeueBlobs(fc_q, tag_in)
                fc = model.net.FC([tag_in, tag_fc_w, tag_fc_b], tag_fc_y,
                                  engine=self.args.engine,
                                  max_num_tasks=self.args.fc_workers)
            else:
                fc = model.net.FC([tag_in, tag_fc_w, tag_fc_b], tag_fc_y,
                                  engine=self.args.engine,
                                  max_num_tasks=self.args.fc_workers)

            layers.append(fc)

            if i == sigmoid_layer:
                layer = model.net.Sigmoid(tag_fc_y, tag_fc_z)

            else:
                layer = model.net.Relu(tag_fc_y, tag_fc_z)
            tag_in = tag_fc_z
            layers.append(layer)

        # WARNING: the dependency between layers is implicit in the tags,
        # so only the last layer is added to the layers list. It will
        # later be used for interactions.
        return layers, weights

    def create_emb(self, m, ln, model, tag, id_qs = None, len_qs = None):
        (tag_layer, tag_in, tag_out) = tag
        emb_l = []
        weights_l = []
        for i in range(0, ln.size):
            n = ln[i]

            # create tags
            len_s = tag_layer + ":::" + "sls" + str(i) + "_l"
            ind_s = tag_layer + ":::" + "sls" + str(i) + "_i"
            tbl_s = tag_layer + ":::" + "sls" + str(i) + "_w"
            sum_s = tag_layer + ":::" + "sls" + str(i) + "_z"
            weights_l.append(tbl_s)

            # initialize the weights
            # approach 1a: custom
            W = np.random.uniform(low=-np.sqrt(1 / n),
                                  high=np.sqrt(1 / n),
                                  size=(n, m)).astype(np.float32)

            # approach 1b: numpy rand
            # W = ra.rand(n, m).astype(np.float32)
            # with core.DeviceScope(core.DeviceOption(caffe2_pb2.CUDA, 0)):
            self.FeedBlobWrapper(tbl_s, W)
            if self.args.queue:
                # If want to have non-blocking IDs we have to dequue the input
                # ID blobs on the model side
                model.net.DequeueBlobs(id_qs[i], ind_s + "_pre_cast")
                model.net.Cast(ind_s + "_pre_cast", ind_s,
                               to=core.DataType.INT32)
                # Operator Mod is not found in Caffe2 latest build
                # model.net.Mod(ind_s + "_pre_mod", ind_s, divisor = n)

                # Dequeue lengths vector as well
                model.net.DequeueBlobs(len_qs[i], len_s)

            # create operator
            if self.gpu_en:
                with core.DeviceScope(core.DeviceOption(caffe2_pb2.CUDA, 0)):
                    EE = model.net.SparseLengthsSum([tbl_s, ind_s, len_s], [sum_s],
                        engine=self.args.engine,
                        max_num_tasks=self.args.sls_workers)
            else:
                EE = model.net.SparseLengthsSum([tbl_s, ind_s, len_s], [sum_s],
                    engine=self.args.engine,
                    max_num_tasks=self.args.sls_workers)
            emb_l.append(EE)

        return emb_l, weights_l

    def create_interactions(self, x, ly, model, tag):
        (tag_dense_in, tag_sparse_in, tag_int_out) = tag

        if self.arch_interaction_op == "dot":
            # concatenate dense and sparse features
            tag_int_out_info = tag_int_out + "_info"
            T, T_info = model.net.Concat(
                x + ly,
                [tag_int_out + "_cat_axis0", tag_int_out_info + "_cat_axis0"],
                axis=1,
                add_axis=1,
            )
            # perform a dot product
            Z = model.net.BatchMatMul([T, T], tag_int_out + "_matmul", trans_b=1)
            # append dense feature with the interactions (into a row vector)
            # approach 1: all
            # Zflat = model.net.Flatten(Z, tag_int_out + "_flatten", axis=1)
            # approach 2: unique
            Zflat_all = model.net.Flatten(Z, tag_int_out + "_flatten_all", axis=1)
            Zflat = model.net.BatchGather([Zflat_all, tag_int_out +"_tril_indices"],
                                           tag_int_out + "_flatten")
            R, R_info = model.net.Concat(
                x + [Zflat], [tag_int_out, tag_int_out_info], axis=1
            )
        elif self.arch_interaction_op == "cat":
            # concatenation features (into a row vector)
            tag_int_out_info = tag_int_out + "_info"
            R, R_info = model.net.Concat(
                x + ly, [tag_int_out, tag_int_out_info], axis=1
            )
        else:
            sys.exit("ERROR: --arch-interaction-op="
                     + self.arch_interaction_op + " is not supported")

        return R

    def create_sequential_forward_ops(self, id_qs = None, len_qs = None, fc_q = None):
        # embeddings
        tag = (self.temb, self.tsin, self.tsout)
        if self.args.sls_type == 'dram':
            self.emb_l, self.emb_w = self.create_emb(self.m_spa, self.ln_emb,
                                                     self.model, tag,
                                                     id_qs = id_qs,
                                                     len_qs = len_qs)
        else:
            m = self.m_spa
            self.emb_l = []
            for i in range(len(self.ln_emb)):
                n = self.ln_emb[i]

                (tag_layer, tag_in, tag_out) = tag
                sum_s = tag_layer + ":::" + "sls" + str(i) + "_z"
                self.FeedBlobWrapper(sum_s, np.array([[0]]).astype(np.float32))
                self.emb_l += [core.BlobReference(sum_s)]

                self.W = np.random.uniform(low=-np.sqrt(1 / n),
                                      high=np.sqrt(1 / n),
                                      size=(n, m)).astype(np.float32)
#               FlatTableType = (c_float * (m * n))
#               self.libFlashRec.unvme_write_table(FlatTableType(*self.W.flatten().tolist()),
#                                                  c_int(m), c_int(n), i)

        # bottom mlp
        tag = (self.tbot, self.tdin, self.tdout)
        self.bot_l, self.bot_w = self.create_mlp(self.ln_bot, self.sigmoid_bot,
                                                 self.model, tag, fc_q = fc_q)
        # interactions
        tag = (self.tdout, self.tsout, self.tint)
        Z = self.create_interactions([self.bot_l[-1]], self.emb_l, self.model, tag)

        # top mlp
        tag = (self.ttop, Z, self.tout)
        self.top_l, self.top_w = self.create_mlp(self.ln_top, self.sigmoid_top,
                                                 self.model, tag
        )

        # setup the last output variable
        self.last_output = self.top_l[-1]

    def __init__(
        self,
        cli_args,
        model=None,
        tag=None,
        enable_prof=False,
        id_qs = None,
        len_qs = None,
        fc_q  = None,
        libFlashRec = None,
    ):
        super(DLRM_Net, self).__init__()
        self.args = cli_args

        self.libFlashRec = libFlashRec

        ### parse command line arguments ###
        ln_bot = np.fromstring(cli_args.arch_mlp_bot, dtype=int, sep="-")
        m_den = ln_bot[0]

        m_spa = cli_args.arch_sparse_feature_size
        ln_emb = np.fromstring(cli_args.arch_embedding_size, dtype=int, sep="-")
        num_fea = ln_emb.size + 1  # num sparse + num dense features
        m_den_out = ln_bot[ln_bot.size - 1]

        gpu_en = self.args.use_gpu

        if cli_args.arch_interaction_op == "dot":
            # approach 1: all
            # num_int = num_fea * num_fea + m_den_out
            # approach 2: unique
            if cli_args.arch_interaction_itself:
                num_int = (num_fea * (num_fea + 1)) // 2 + m_den_out
            else:
                num_int = (num_fea * (num_fea - 1)) // 2 + m_den_out
        elif cli_args.arch_interaction_op == "cat":
            num_int = num_fea * m_den_out
        else:
            sys.exit("ERROR: --arch-interaction-op="
                     + cli_args.arch_interaction_op + " is not supported")

        arch_mlp_top_adjusted = str(num_int) + "-" + cli_args.arch_mlp_top
        ln_top = np.fromstring(arch_mlp_top_adjusted, dtype=int, sep="-")

        if m_den != ln_bot[0]:
            sys.exit("ERROR: arch-dense-feature-size "
                + str(m_den) + " does not match first dim of bottom mlp " + str(ln_bot[0]))
        if m_spa != m_den_out:
            sys.exit("ERROR: arch-sparse-feature-size "
                + str(m_spa) + " does not match last dim of bottom mlp " + str(m_den_out))
        if num_int != ln_top[0]:
            sys.exit("ERROR: # of feature interactions "
                + str(num_int) + " does not match first dim of top mlp " + str(ln_top[0]))

        ### initialize the model ###
        if model is None:
            global_init_opt = ["caffe2", "--caffe2_log_level=0"]
            if enable_prof:
                global_init_opt += [
                    "--logtostderr=0",
                    "--log_dir=$HOME",
                    #"--caffe2_logging_print_net_summary=1",
                ]
            workspace.GlobalInit(global_init_opt)
            self.set_tags()
            self.model = model_helper.ModelHelper(name="DLRM", init_params=True)

            if cli_args:
              self.model.net.Proto().type = cli_args.caffe2_net_type
              self.model.net.Proto().num_workers = cli_args.inter_op_workers

        else:
            # WARNING: assume that workspace and tags have been initialized elsewhere
            self.set_tags(tag[0], tag[1], tag[2], tag[3], tag[4], tag[5], tag[6],
                          tag[7], tag[8], tag[9])
            self.model = model

        # save arguments
        self.m_spa = m_spa
        self.ln_emb = ln_emb
        self.ln_bot = ln_bot
        self.ln_top = ln_top
        self.arch_interaction_op = cli_args.arch_interaction_op
        self.arch_interaction_itself = cli_args.arch_interaction_itself
        self.sigmoid_bot = -1 # TODO: Lets not hard-code this going forward
        self.sigmoid_top = ln_top.size - 1
        self.gpu_en = gpu_en

        return self.create_sequential_forward_ops(id_qs, len_qs, fc_q)

    def set_tags(
        self,
        _tag_layer_top_mlp="top",
        _tag_layer_bot_mlp="bot",
        _tag_layer_embedding="emb",
        _tag_feature_dense_in="dense_in",
        _tag_feature_dense_out="dense_out",
        _tag_feature_sparse_in="sparse_in",
        _tag_feature_sparse_out="sparse_out",
        _tag_interaction="interaction",
        _tag_dense_output="prob_click",
        _tag_dense_target="target",
    ):
        # layer tags
        self.ttop = _tag_layer_top_mlp
        self.tbot = _tag_layer_bot_mlp
        self.temb = _tag_layer_embedding
        # dense feature tags
        self.tdin = _tag_feature_dense_in
        self.tdout = _tag_feature_dense_out
        # sparse feature tags
        self.tsin = _tag_feature_sparse_in
        self.tsout = _tag_feature_sparse_out
        # output and target tags
        self.tint = _tag_interaction
        self.ttar = _tag_dense_target
        self.tout = _tag_dense_output

    def parameters(self):
        return self.model

    def create(self, X, S_lengths, S_indices, T, id_qs = None, len_qs=None):
        self.create_input(X, S_lengths, S_indices, T)
        self.create_model(X, S_lengths, S_indices, T)

    def create_input(self, X, S_lengths, S_indices, T):
        # feed input data to blobs
        self.FeedBlobWrapper(self.tdin, X)

        for i in range(len(self.emb_l)):
            len_s = self.temb + ":::" + "sls" + str(i) + "_l"
            ind_s = self.temb + ":::" + "sls" + str(i) + "_i"
            self.FeedBlobWrapper(len_s, np.array(S_lengths[i]))
            self.FeedBlobWrapper(ind_s, np.array(S_indices[i]))

        # feed target data to blobs
        if T is not None:
            zeros_fp32 = np.zeros(T.shape).astype(np.float32)
            self.FeedBlobWrapper(self.ttar, zeros_fp32)


    def create_model(self, X, S_lengths, S_indices, T):
        #setup tril indices for the interactions
        offset = 1 if self.arch_interaction_itself else 0
        num_fea = len(self.emb_l) + 1
        tril_indices = np.array([j + i * num_fea
                                 for i in range(num_fea) for j in range(i + offset)])
        self.FeedBlobWrapper(self.tint + "_tril_indices", tril_indices)

        # create compute graph
        print("Trying to run DLRM for the first time")
        sys.stdout.flush()
        if T is not None:
            # WARNING: RunNetOnce call is needed only if we use brew and ConstantFill.
            # We could use direct calls to self.model functions above to avoid it
            workspace.RunNetOnce(self.model.param_init_net)
            workspace.CreateNet(self.model.net)
        print("Ran DLRM for the first time")
        sys.stdout.flush()


    def run(self, X=None, S_lengths=None, S_indices=None, listResult=None, enable_prof=False):
        # feed input data to blobs
        if not self.args.queue:
            # dense features
            self.FeedBlobWrapper(self.tdin, X)

            ## sparse features
            if self.args.sls_type == 'dram':
                for i in range(len(self.emb_l)):
                    ind_s = self.temb + ":::" + "sls" + str(i) + "_i"
                    self.FeedBlobWrapper(ind_s, np.array(S_indices[i]))

                    len_s = self.temb + ":::" + "sls" + str(i) + "_l"
                    self.FeedBlobWrapper(len_s, np.array(S_lengths[i]))
            elif ((self.args.sls_type == 'base') or
                 (self.args.sls_type == 'ndp')):
                for i in range(len(S_lengths)):
                    sum_s = self.temb + ":::" + "sls" + str(i) + "_z"
                    self.FeedBlobWrapper(sum_s, np.array(listResult).astype(np.float32))

        # execute compute graph
        if enable_prof:
            workspace.C.benchmark_net(self.model.net.Name(), 0, 1, True)
        else:
            workspace.RunNet(self.model.net)


def slsload(sls_type, lru, libFlashRec, m_spa, X, S_lengths, S_indices, qid, ln_emb, mini_batch_size, indices_per_lookup):
    slsload_enter = time.time()
    global slsloadtime
    global unvmelibtime
    global unvmetranslatetime
    global hits
    global misses
    localhits = 0
    localmisses = 0
    ## sparse features
    listResult = None
    if sls_type == 'base':
        for i in xrange(len(ln_emb)):
            batchsize = mini_batch_size
            embed_per_result = indices_per_lookup

            listResult = []
            for j in xrange(batchsize):
                embedding = np.ones(m_spa)
                for k in xrange(embed_per_result):
                    embedidx = S_indices[i][j * embed_per_result + k]
                    libFlashRec.unvme_read_embedding.restype = POINTER(c_float)
                    read_embedding = []
                    if (lru[qid].has_key(embedidx)):
                        localhits += 1
                        hits += 1
                        read_embedding = lru[qid][embedidx]
                    else:
                        localmisses += 1
                        misses += 1

                        unvme_enter = time.time()

                        read_embedding = libFlashRec.unvme_read_embedding(
                            c_int(embedidx), c_int(m_spa), c_int(i), c_int(qid))

                        unvme_exit = time.time()

                        list_embedding = []
                        for h in xrange(m_spa+1):
                            list_embedding.append(read_embedding[h])
                        read_embedding = np.array(list_embedding).astype(np.float32)
                        lru[qid][embedidx] = read_embedding[:m_spa]
                        if (read_embedding[m_spa] > 0):
                            unvmelibtime[qid] += (unvme_exit - unvme_enter)
                            unvmetranslatetime[qid] += (read_embedding[m_spa])

                    embedding = np.multiply(embedding, read_embedding[:m_spa])
                listResult.append(embedding)

            listResult = np.array(listResult).astype(np.float32)
    elif sls_type == 'ndp':
        for i in range(len(ln_emb)):
            batchsize = mini_batch_size
            embed_per_result = indices_per_lookup
            num_lookups = mini_batch_size*indices_per_lookup

            split = [S_indices[i][:num_lookups]]
            if batchsize > 128:
                split = np.split(np.array(S_indices[i][:num_lookups]), (batchsize/128))
                batchsize = 128
            listResult = []
            for indices in split:
                pairInd = zip((j for k in xrange(embed_per_result) for j in xrange(batchsize)),
                              indices)
                if (0):
                    numInputEmbeddings = len(pairInd)
                    flatInd = np.array(sorted(pairInd, key=(lambda x: x[1]))).flatten()
                    #print("Printing flatInd for table"+str(i)+" and q"+str(qid))
                    #print("batchsize="+str(batchsize)+", embed_per_result="+str(embed_per_result))
                    #print("shape="+str(flatInd.shape))
                    #print(flatInd)

                    IndexType = c_int * len(flatInd)
                    #sys.exit()

                    unvme_enter = time.time()

                    libFlashRec.unvme_sparse_length_sum.restype = POINTER(c_float)

                    unvmelibtime[qid] += (time.time() - unvme_enter)

                    result = libFlashRec.unvme_sparse_length_sum(
                            IndexType(*flatInd),
                            c_int(m_spa), c_int(batchsize), c_int(embed_per_result), c_int(i), c_int(qid), c_int(numInputEmbeddings))
                    for k in xrange(batchsize):
                        listResult.append([result[k*m_spa+j] for j in xrange(m_spa)])
                    listResult = np.array(listResult).astype(np.float32)

                    unvmetranslatetime[qid] += (result[batchsize*m_spa])
                else:
                    filteredPairInd = []
                    cachedEmbPair = []
                    for (resultidx, embedidx) in pairInd:
                        if embedidx in lru[qid]:
                            localhits += 1
                            hits += 1
                            cachedEmbPair.append((resultidx, lru[qid][embedidx]))
                        else:
                            localmisses += 1
                            misses += 1
                            filteredPairInd.append((resultidx, embedidx))

                    # static partitioning
                    numInputEmbeddings = len(filteredPairInd)
                    flatInd = np.array(sorted(filteredPairInd, key=(lambda x: x[1]))).flatten()
                    IndexType = c_int * len(flatInd)
                    libFlashRec.unvme_sparse_length_sum.restype = POINTER(c_float)
                    result = libFlashRec.unvme_sparse_length_sum(
                            IndexType(*flatInd),
                            c_int(m_spa), c_int(batchsize), c_int(embed_per_result), c_int(i), c_int(qid), c_int(numInputEmbeddings))
                    for k in xrange(batchsize):
                        listResult.append([result[k*batchsize+j] for j in xrange(m_spa)])
                    listResult = np.array(listResult).astype(np.float32)

                    for (resultidx, embed) in cachedEmbPair:
                        listResult[resultidx] = np.multiply(listResult[resultidx], embed)


    if (0):
        localhitrate = 0
    else:
        localhitrate = (float(localhits)/float(localhits+localmisses))*100.
    slsloadtime[qid] += (time.time() - slsload_enter)
    return (listResult, localhitrate)


if __name__ == "__main__":
    ### import packages ###
    import sys
    import argparse

    sys.path.append("..")
    # data generation
    from data_generator.dlrm_data_caffe2 import DLRMDataGenerator

    from utils.utils import cli

    args = cli()

    ### some basic setup ###
    np.random.seed(args.numpy_rand_seed)
    np.set_printoptions(precision=args.print_precision)

    use_gpu = args.use_gpu
    if use_gpu:
        device_opt = core.DeviceOption(caffe2_pb2.CUDA, 0)
        ngpus = C.num_cuda_devices  # 1
        print("Using {} GPU(s)...".format(ngpus))
    else:
        device_opt = core.DeviceOption(caffe2_pb2.CPU)
        print("Using CPU...")

    ### prepare training data ###
    ln_bot = np.fromstring(args.arch_mlp_bot, dtype=int, sep="-")
    dc = DLRMDataGenerator (args)
    if args.data_generation == "dataset":
        print("Error we have disabled this function currently....")
        sys.exit()
        # input and target data
        #(nbatches, lX, lS_l, lS_i, lT,
        # nbatches_test, lX_test, lS_l_test, lS_i_test, lT_test,
        # ln_emb, m_den) = dc.read_dataset(
        #    args.data_set, args.mini_batch_size, args.data_randomize, args.num_batches,
        #    True, args.raw_data_file, args.processed_data_file)
        #ln_bot[0] = m_den
    else:
        # input data
        ln_emb = np.fromstring(args.arch_embedding_size, dtype=int, sep="-")
        m_den = ln_bot[0]
        if args.data_generation == "random":
            (nbatches, lX, lS_l, lS_i) = dc.generate_input_data()
        elif args.data_generation == "synthetic":
            (nbatches, lX, lS_l, lS_i) = dc.generate_synthetic_input_data(
                args.num_batches, args.mini_batch_size,
                args.round_targets, args.num_indices_per_lookup,
                args.num_indices_per_lookup_fixed, m_den, ln_emb,
                args.data_trace_file, args.data_trace_enable_padding)
        elif args.data_generation == "load_data":
            with open(args.data_file) as f:
                (nbatches, lX, lS_l, lS_i)  = pickle.load(f)
        else:
            sys.exit("ERROR: --data-generation="
                     + args.data_generation + " is not supported")
        # target data
        print("Generating output dataset")
        (nbatches, lT) = dc.generate_output_data()

    num_lookups_per_batch = args.mini_batch_size*args.num_indices_per_lookup
    for j in xrange(nbatches):
        lX[j] = lX[j][:args.mini_batch_size]
        for k in xrange(len(ln_emb)):
            lS_l[j][k] = lS_l[j][k][:args.mini_batch_size]
            lS_i[j][k] = lS_i[j][k][:num_lookups_per_batch]

    ### construct the neural network specified above ###
    print("Trying to initialize DLRM")
    libFlashRec = cdll.LoadLibrary("./libflashrec.so")
    libFlashRec.open_unvme()
    print("libFlashRec opened")
    load_instances = 8
    run_instances = 1
    dlrm_run_instances = []
    with core.DeviceScope(device_opt):
        for i in xrange(run_instances):
            dlrm_run_instances.append(DLRM_Net( args, libFlashRec=libFlashRec ))
    print("Initialized DLRM Net")
    for dlrm in dlrm_run_instances:
        dlrm.create(
                lX[0],
                lS_l[0],
                lS_i[0],
                lT[0])
    print("Created network")

    global slsloadtime
    global unvmelibtime
    global unvmetranslatetime
    slsloadtime = []
    unvmelibtime = []
    unvmetranslatetime = []
    for i in xrange(load_instances):
        slsloadtime.append(0)
        unvmelibtime.append(0)
        unvmetranslatetime.append(0)

    lru = []
    global hits
    global misses
    hits = 0
    misses = 0
    if args.sls_type != 'dram':
        if (0):
            for i in xrange(load_instances):
                lru.append(LRU(2000))
        else:
            # Static partitioning
            for i in xrange(load_instances):
                lru.append(LRU(2000))
            for i in xrange(load_instances):
                profile = {}
                for b in xrange(nbatches):
                    for index in lS_i[b][i][:num_lookups_per_batch]:
                        if index in profile:
                            profile[index] += 1
                        else:
                            profile[index] = 1
                for (k, (key, value)) in enumerate(sorted(profile.items(),
                                                          key=lambda x: x[1],
                                                          reverse=True)):
                    lru[i][key] = dlrm_run_instances[0].W[key]
                    if k == 2000:
                        break

    total_time = 0
    dload_time = 0
    k = 0

    #exit(0)
    batchtime_start = np.zeros(nbatches)
    batchtime_end = np.zeros(nbatches)
    batchhitrate = np.zeros(nbatches)

    time_start = time.time()

    print("Running networks")
    def stage_run_dlrm(dlrm, run_q, stop):
        while True:
            try:
                (batch, listResult) = run_q.get(block=False)
                # forward and backward pass, where the latter runs only
                # when gradients and loss have been added to the net
                dlrm.run(lX[batch], lS_l[batch], lS_i[batch], listResult)
                run_q.task_done()
            except Queue.Empty:
                if stop():
                    break
                else:
                    continue
    def stage_load_dlrm(run_q, batch_q, qid, stop, sls_type, lru, libFlashRec, sparse_feature_size):
        while True:
            try:
                batch = batch_q.get(block=False)
                #print("Loading batch " + str(batch))
                (listResult, localhitrate) = slsload(sls_type, lru, libFlashRec, sparse_feature_size, lX[batch], lS_l[batch], lS_i[batch], qid, ln_emb, args.mini_batch_size, args.num_indices_per_lookup)
                run_q.put((batch, listResult))
                batch_q.task_done()
            except Queue.Empty:
                if stop():
                    break
                else:
                    continue
    stop_workers = False
    run_q = Queue.Queue()
    batch_q = Queue.Queue()
    for i in xrange(run_instances):
        run_worker = Thread(target = stage_run_dlrm,
                args = (dlrm_run_instances[i], run_q, (lambda : stop_workers)))
        run_worker.setDaemon(True)
        run_worker.start()
    for i in xrange(load_instances):
        load_worker = Thread(target = stage_load_dlrm,
                args = (run_q, batch_q, i, (lambda : stop_workers),
                    args.sls_type, lru, libFlashRec, args.arch_sparse_feature_size))
        load_worker.setDaemon(True)
        load_worker.start()
    for k in xrange(args.nepochs):
        for i in xrange(nbatches):
            batchtime_start[i] = time.time()

            (listResult, localhitrate) = slsload(args.sls_type, lru, libFlashRec, args.arch_sparse_feature_size, lX[i], lS_l[i], lS_i[i], 0, ln_emb, args.mini_batch_size, args.num_indices_per_lookup)
            dlrm.run(lX[i], lS_l[i], lS_i[i], listResult)
            #batch_q.put(i)

            batchhitrate[i] = localhitrate
            batchtime_end[i] = time.time()
    batch_q.join()
    run_q.join()
    print("All done here, cleaning up")
    if ((hits+misses) != 0):
        print("Cache hitrate: "+str(int((float(hits)/float(hits+misses))*100)))
    stop_workers = True
    libFlashRec.flush_unvme()
    libFlashRec.close_unvme()

    time_end = time.time()
    dload_time *= 1000.
    total_time += (time_end - time_start) * 1000.

    slsloadtime = sum(slsloadtime) * 1000.
    unvmelibtime = sum(unvmelibtime) * 1000.
    unvmetranslatetime = sum(unvmetranslatetime) * 1000.
    print("Total SLS load time: ***", slsloadtime, " ms")
    print("Total UNVME lib time: ***", unvmelibtime, " ms")
    print("Total UNVME translation time: ***", unvmetranslatetime, " ms")

    print("Total data loading time: ***", dload_time, " ms")
    print("Total data loading time: ***", dload_time / (args.nepochs * nbatches), " ms/iter")
    print("Total computation time: ***", (total_time - dload_time), " ms")
    print("Total computation time: ***", (total_time - dload_time) / (args.nepochs * nbatches), " ms/iter")
    print("Total execution time: ***", total_time, " ms")
    print("Total execution time: ***", total_time / (args.nepochs * nbatches), " ms/iter")


    batchtime = ((batchtime_end - batchtime_start)*1000.)

    fig, ax = plt.subplots(tight_layout=True)
    N, bins, patches = ax.hist(batchtime, bins=100)

    pickle.dump( (batchtime, batchhitrate) , open( "batchtimes.p", "wb" ) )

    timehitpairs = sorted(zip(batchtime, batchhitrate), key=(lambda x: x[0]))
    binhitrates = []
    for b in xrange(bins.size-1):
        binhitrates.append([])
    histhitrate = np.zeros(bins.size-1)
    i = 0
    for b in xrange(bins.size-1):
        while i < nbatches and timehitpairs[i][0] < bins[b+1]:
            binhitrates[b].append(timehitpairs[i][1])
            i += 1
        histhitrate[b] = np.mean(binhitrates[b])

    filteredhitrate = np.array([x for x in histhitrate if math.isnan(x) == False])
    norm = colors.Normalize(filteredhitrate.min(), filteredhitrate.max())
    for hitrate, patch in zip(histhitrate, patches):
        color = plt.cm.viridis(norm(hitrate))
        patch.set_facecolor(color)

    plt.savefig("batchtimes.png")
    print(batchtime)
    print(batchhitrate)
    print(histhitrate)
    print(filteredhitrate.min())
    print(filteredhitrate.max())

    print("P99: " + str(sorted(batchtime)[int(len(batchtime)*0.99)]))
    print("P90: " + str(sorted(batchtime)[int(len(batchtime)*0.90)]))
    print("P50: " + str(sorted(batchtime)[int(len(batchtime)*0.50)]))

    sys.exit()
