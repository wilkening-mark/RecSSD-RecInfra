from __future__ import absolute_import, division, print_function, unicode_literals

from multiprocessing import Queue, Process
from utils.packets   import ServiceRequest
from utils.utils  import debugPrint
import time
import numpy as np
import sys
import math
#from simple_pid import PID

def model_arrival_times(args):
  arrival_time_delays = np.random.poisson(lam  = args.avg_arrival_rate,
                                          size = args.nepochs * args.num_batches)
  return arrival_time_delays


def model_batch_size_distribution(args):
  if args.batch_size_distribution == "normal":
    batch_size_distributions = np.random.normal(args.avg_mini_batch_size, args.var_mini_batch_size, args.num_batches)

  elif args.batch_size_distribution == "lognormal":
    batch_size_distributions = np.random.lognormal(args.avg_mini_batch_size, args.var_mini_batch_size, args.num_batches)

  elif args.batch_size_distribution == "fixed":
    batch_size_distributions = np.array([args.mini_batch_size for _ in range(args.num_batches) ])

  elif args.batch_size_distribution == "file":
    percentiles = []
    batch_size_distributions = []
    with open(args.batch_dist_file, 'r') as f:
      lines = f.readlines()
      for line in lines:
        percentiles.append(float(line.rstrip()))

      for _ in range(args.num_batches):
        batch_size_distributions.append( int(percentiles[ int(np.random.uniform(0, len(percentiles))) ]) )

  for i in range(args.num_batches):
    batch_size_distributions[i] = int(max(min(batch_size_distributions[i], args.max_mini_batch_size), 1))
  return batch_size_distributions


def partition_requests(args, batch_size):
  batch_sizes = []

  while batch_size > 0:
    mini_batch_size = min(args.sub_task_batch_size, batch_size)
    batch_sizes.append(mini_batch_size)
    batch_size -= mini_batch_size

  return batch_sizes

def loadGenSleep( sleeptime ):
  if sleeptime > 0.0055:
    time.sleep(sleeptime)
  else:
    startTime = time.time()
    while (time.time() - startTime) < sleeptime:
      continue
  return

def HillClimbing(args,
                pidQueue,
                arr_id,
                tried_arrival_rates,
                qps_tried,
                batch_config_qps,
                batch_config_attempt,
                batch_configs,
                requestQueue,
                gpuRequestQueue,
                tuning_batch_qps,
                tuning_gpu_qps):

  ########################################################################
  running_latency = pidQueue.get()

  minarr = args.min_arr_range
  maxarr = args.max_arr_range
  steps  = args.arr_steps

  possible_arrival_rates = np.logspace(math.log(minarr, 10), math.log(maxarr, 10), num=steps)

  look_back       = args.look_back
  unique_arrivals = args.unique_arrivals
  ########################################################################

  if running_latency > args.target_latency * (1 + args.stable_region):
    # if running latency is too high then we increase inter-arrival time (decrease QPS)
    arr_id = min( len(possible_arrival_rates) - 1, arr_id + 1)
  elif running_latency >= args.target_latency:
    arr_id = arr_id
  elif running_latency < args.target_latency / (1 + args.stable_region):
    # if running latency is too low then we increase inter-arrival time (decrease QPS)
    arr_id = max( 0, arr_id - 1)
  else:
    # if running latency is too low then we increase inter-arrival time (decrease QPS)
    arr_id = arr_id

  arrival_rate =  possible_arrival_rates[arr_id]

  tried_arrival_rates.append(arrival_rate)

  qps_tried += 1

  if qps_tried > args.sched_timeout:
    # System configuration is unstable and thus we will assume worst
    # case from the last `timeout` batches
    arrival_rate = np.median(tried_arrival_rates[-1 * args.arr_steps:])
    print("Found fixed arrival rate:::", arrival_rate, "ms")

    batch_config_qps.append(arrival_rate)
    batch_config_attempt += 1

    #if (len(batch_config_qps) >= 2) and (batch_config_qps[-1] > batch_config_qps[-2]):
    #  # Given that increasing the batch size configuration does not
    #  # help we set the optimal configuration as the previous one
    #  # tried

    #  # We have found optimal configuration
    #  arrival_rate =  batch_config_qps[batch_config_attempt - 2]
    #  qps_tried = 0
    #  if tuning_batch_qps:
    #    args.sub_task_batch_size = batch_configs[batch_config_attempt - 2]
    #    tuning_batch_qps = False
    #    print("[found opt] Optimal batch_size configuration: ", args.sub_task_batch_size, " @ arrival rate of ", arrival_rate, "ms")
    #  elif tuning_gpu_qps:
    #    args.gpu_request_size_thres = batch_configs[batch_config_attempt - 2]
    #    tuning_gpu_qps = False
    #    print("[found opt] Optimal gpu configuration: ", args.gpu_request_size_thres, " @ arrival rate of ", arrival_rate, "ms")

    if (len(batch_config_qps) == len(batch_configs)):
      # If we tried all possible configurations and the last one was
      # not worse than the 2nd to last then we know that the optimal configuration is the last one

      # We have found optimal configuration
      arrival_rate = min(batch_config_qps)
      best_attempt = np.argmin(batch_config_qps)

      qps_tried = 0
      if tuning_batch_qps:
        args.sub_task_batch_size = batch_configs[best_attempt]
        tuning_batch_qps = False
        print("[tried all ] Optimal batch_size configuration: ", args.sub_task_batch_size, " @ arrival rate of ", arrival_rate, "ms")
      elif tuning_gpu_qps:
        args.gpu_request_size_thres = batch_configs[best_attempt]
        tuning_gpu_qps = False
        print("[tried all ] Optimal gpu configuration: ", args.gpu_request_size_thres, " @ arrival rate of ", arrival_rate, "ms")

    else:
      # Else we find that that the achievable QPS has gone up so we
      # need to keep trying optimal batch-size configurations. We
      # should not be equal to the length  of batch_configs as that
      # would have been caught by the previous condition
      if tuning_batch_qps:
        args.sub_task_batch_size = batch_configs[batch_config_attempt]
      elif tuning_gpu_qps:
        args.gpu_request_size_thres = batch_configs[batch_config_attempt]

      # Need to try the next batch-size configuration so have to
      # reset the hill-climbing
      tried_arrival_rates = []
      qps_tried = 0
      arrival_rate = args.avg_arrival_rate
      arr_id = np.argmin(np.abs(possible_arrival_rates-args.avg_arrival_rate))

    # Drain the request and and wait for the next iteration queue
    while requestQueue.qsize() > 500:
      requestQueue.get()

    # Drain the request and and wait for the next iteration queue
    while gpuRequestQueue.qsize() > 500:
      gpuRequestQueue.get()

    # Drain the request and and wait for the next iteration queue
    while requestQueue.qsize() > 0:
      continue

    # Drain the request and and wait for the next iteration queue
    while gpuRequestQueue.qsize() > 0:
      continue

    time.sleep(3)

    while pidQueue.qsize() > 0:
      pidQueue.get()

  else:
    print("New arrival rate:::", arrival_rate, "ms")

  sys.stdout.flush()

  out = (args, arrival_rate, arr_id, tried_arrival_rates, batch_config_qps, batch_config_attempt, tuning_batch_qps, tuning_gpu_qps, qps_tried)

  return out


def loadGenerator(args,
                  requestQueue,
                  loadGeneratorReturnQueue,
                  inferenceEngineReadyQueue,
                  pidQueue,
                  gpuRequestQueue):

  debugPrint(args, "Load Generator", "Number of total inference engines" + str(args.inference_engines))
  ready_engines = 0

  while ready_engines < args.inference_engines:
    inferenceEngineReadyQueue.get()
    ready_engines += 1

  arrival_time_delays = model_arrival_times(args)

  batch_size_distributions = model_batch_size_distribution(args)

  cpu_sub_requests = 0
  cpu_requests = 0
  gpu_requests = 0

  minarr = args.min_arr_range
  maxarr = args.max_arr_range
  steps  = args.arr_steps

  possible_arrival_rates = np.logspace(math.log(minarr, 10), math.log(maxarr, 10), num=steps)
  arr_id = np.argmin(np.abs(possible_arrival_rates-args.avg_arrival_rate))

  print("Arrival rates to try: ", possible_arrival_rates)

  # Instantiate pid controller
  arrival_rate = args.avg_arrival_rate
  tried_arrival_rates = []
  fixed_arrival_rate = None

  tuning_batch_qps = args.tune_batch_qps
  tuning_gpu_qps   = False

  batch_configs = np.fromstring(args.batch_configs, dtype=int, sep="-")
  batch_config_attempt = 0

  gpu_configs = np.fromstring(args.gpu_configs, dtype=int, sep="-")
  gpu_config_attempt = 0

  if tuning_batch_qps:
    args.sub_task_batch_size = batch_configs[batch_config_attempt]

  # To start with lets not run with the GPU sweeps
  args.gpu_request_size_thres = 1024

  batch_config_qps = []
  gpu_config_qps   = []

  epoch = 0
  exp_epochs = 0
  qps_tried = 0

  while tuning_batch_qps or (exp_epochs < args.nepochs):
    for batch_id in range(args.num_batches):
      # absolute request ID
      request_id = epoch * args.num_batches + batch_id

      # ####################################################################
      # Batch size hill climbing
      # ####################################################################
      if tuning_batch_qps and (pidQueue.qsize() > 0):
        out = HillClimbing(args,
                           pidQueue,
                           arr_id,
                           tried_arrival_rates,
                           qps_tried,
                           batch_config_qps,
                           batch_config_attempt,
                           batch_configs,
                           requestQueue,
                           gpuRequestQueue,
                           tuning_batch_qps,
                           tuning_gpu_qps)

        args, arrival_rate, arr_id, tried_arrival_rates, batch_config_qps, batch_config_attempt, tuning_batch_qps, tuning_gpu_qps, qps_tried = out

        if tuning_batch_qps == False:
          print("Finished batch size scheduler ")
          # Reset the hill climbing parameters for gpu hill climbing
          qps_tried = 0
          arr_id = np.argmin(np.abs(possible_arrival_rates-args.avg_arrival_rate))
          tried_arrival_rates = []
          gpu_config_qps = []
          gpu_config_attempt = 0

          gpu_configs = np.fromstring(args.gpu_configs, dtype=int, sep="-")

          if args.model_gpu and args.tune_gpu_qps:
            print("Starting gpu scheduler")
            tuning_gpu_qps = True

      # ####################################################################
      # GPU partition hill climbing
      # ####################################################################
      if args.model_gpu and tuning_gpu_qps and (pidQueue.qsize() > 0):
        out = HillClimbing(args,
                           pidQueue,
                           arr_id,
                           tried_arrival_rates,
                           qps_tried,
                           gpu_config_qps,
                           gpu_config_attempt,
                           gpu_configs,
                           requestQueue,
                           gpuRequestQueue,
                           tuning_batch_qps,
                           tuning_gpu_qps)

        args, arrival_rate, arr_id, tried_arrival_rates, gpu_config_qps, gpu_config_attempt, tuning_batch_qps, tuning_gpu_qps, qps_tried = out

      request_size = int(batch_size_distributions[batch_id])

      if args.model_gpu and (request_size >= args.gpu_request_size_thres):
        # If request size is larger  than the threshold then we want to send it
        # over to the GPU.
        request = ServiceRequest(batch_id = batch_id,
                                 epoch = epoch,
                                 batch_size = request_size,
                                 sub_id = 0,
                                 total_sub_batches = 1,
                                 exp_packet = (tuning_batch_qps or tuning_gpu_qps) )

        gpu_requests += 1

        # add timestamp for request arriving onto server
        request.arrival_time = time.time()

        debugPrint(args, "Load Generator", "Adding request to GPU")
        gpuRequestQueue.put(request)
        debugPrint(args, "Load Generator", "Added request to GPU")

      else:

        batch_sizes = partition_requests(args, request_size)
        for i, batch_size in enumerate(batch_sizes):
          # create request

          request = ServiceRequest(batch_id = batch_id,
                                   epoch = epoch,
                                   batch_size = batch_size,
                                   sub_id = i,
                                   total_sub_batches = len(batch_sizes),
                                   exp_packet = (tuning_batch_qps or tuning_gpu_qps))
          cpu_sub_requests += 1

          # add timestamp for request arriving onto server
          request.arrival_time = time.time()
          requestQueue.put(request)
        cpu_requests += 1

      arrival_time = np.random.poisson(lam = arrival_rate, size = 1)
      loadGenSleep( arrival_time / 1000.   )

    epoch += 1

    if (tuning_batch_qps == False) and (tuning_gpu_qps == False):
      exp_epochs += 1

  # Signal to the backend consumers that we are done
  for i in range(args.inference_engines):
    if args.model_gpu and (i == (args.inference_engines-1)):
      debugPrint(args, "Load Generator", "sending done signal to " + str(i) + " gpu engine")
      gpuRequestQueue.put(None)
    else:
      debugPrint(args, "Load Generator", "sending done signal to " + str(i) + " cpu engine")
      requestQueue.put(None)

  # Return total number of sub-tasks simulated
  loadGeneratorReturnQueue.put( (cpu_sub_requests, cpu_requests, gpu_requests) )

  return


if __name__=="__main__":
  main()
