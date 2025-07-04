import time
import argparse
import pickle
import os
import datetime
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from utils import *
from modules import *

#os.environ['CUDA_LAUNCH_BLOCKING'] = ""
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'


parser = argparse.ArgumentParser(
    'Variant neral relational inference for molecular dynamics simulations')
parser.add_argument('--num-residues', type=int, default=220,
                    help='Number of residues of the PDB.')
parser.add_argument('--save-folder', type=str, default='logs',
                    help='Where to save the trained model, leave empty to not save anything.')
parser.add_argument('--load-folder', type=str, default='',
                    help='Where to load the trained model if finetunning. ' +
                         'Leave empty to train from scratch')
parser.add_argument('--edge-types', type=int, default=4,
                    help='The number of edge types to infer.')
parser.add_argument('--dims', type=int, default=6,
                    help='The number of input dimensions used in study( position (X,Y,Z) + velocity (X,Y,Z) ). ')
parser.add_argument('--timesteps', type=int, default=50,
                    help='The number of time steps per sample. Actually is 50')
parser.add_argument('--prediction-steps', type=int, default=1, metavar='N',
                    help='Num steps to predict before re-using teacher forcing.')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=100,
                    help='Number of epochs to train.')
parser.add_argument('--batch-size', type=int, default=1,
                    help='Number of samples per batch.')
parser.add_argument('--lr', type=float, default=0.0005,
                    help='Initial learning rate.')
parser.add_argument('--encoder-hidden', type=int, default=256,
                    help='Number of hidden units in encoder.')
parser.add_argument('--decoder-hidden', type=int, default=256,
                    help='Number of hidden units in decoder.')
parser.add_argument('--temp', type=float, default=0.5,
                    help='Temperature for Gumbel softmax.')
parser.add_argument('--encoder', type=str, default='mlpatten',
                    help='Type of path encoder model (mlp , cnn or mlpatten).')
parser.add_argument('--decoder', type=str, default='rnn',
                    help='Type of decoder model (mlp, rnn, or sim).')
parser.add_argument('--no-factor', action='store_true', default=False,
                    help='Disables factor graph model.')
parser.add_argument('--encoder-dropout', type=float, default=0.0,
                    help='Dropout rate (1 - keep probability) in encoder.')
parser.add_argument('--decoder-dropout', type=float, default=0.0,
                    help='Dropout rate (1 - keep probability) in decoder.')
parser.add_argument('--lr-decay', type=int, default=200,
                    help='After how epochs to decay LR by a factor of gamma.')
parser.add_argument('--gamma', type=float, default=0.5,
                    help='LR decay factor.')
parser.add_argument('--skip-first', action='store_true', default=True,
                    help='Skip first edge type in decoder, i.e. it represents no-edge.')
parser.add_argument('--var', type=float, default=5e-5,
                    help='Output variance.')
parser.add_argument('--hard', action='store_true', default=True,
                    help='Uses discrete samples in training forward pass.')
parser.add_argument('--prior', action='store_true', default=True,
                    help='Whether to use sparsity prior.')
parser.add_argument('--dynamic-graph', action='store_true', default=True,
                    help='Whether test with dynamically re-computed graph.')
parser.add_argument('--number-expstart', type=int, default=0,
                    help='start number of experiments.')
parser.add_argument('--number-exp', type=int, default=56,
                    help='number of experiments.')
args = parser.parse_args()
args.cuda = True
# args.cuda = not args.no_cuda and torch.cuda.is_available()
args.factor = not args.no_factor
# print all arguments
print(args)

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

if args.dynamic_graph:
    print("Testing with dynamically re-computed graph.")

# Save model and meta-data. Always saves in a new sub-folder.
if args.save_folder:
    exp_counter = 0
    now = datetime.datetime.now()
    timestamp = now.isoformat()
    save_folder = args.save_folder+'/'
    if not os.path.isdir(save_folder):
        os.mkdir(save_folder)
    meta_file = os.path.join(save_folder, 'metadata.pkl')
    encoder_file = os.path.join(save_folder, 'encoder.pt')
    decoder_file = os.path.join(save_folder, 'decoder.pt')

    log_file = os.path.join(save_folder, 'log.txt')
    log = open(log_file, 'w')

    pickle.dump({'args': args}, open(meta_file, "wb"))
else:
    print("WARNING: No save_folder provided!" +
          "Testing (within this script) will throw an error.")

# load data
train_loader, valid_loader, test_loader, loc_max, loc_min, vel_max, vel_min = load_dataset_train_valid_test(
    args.batch_size, args.number_exp, args.number_expstart, args.dims)

# Generate off-diagonal interaction graph
off_diag = np.ones([args.num_residues, args.num_residues]
                   ) - np.eye(args.num_residues)

rel_rec = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float32)
rel_send = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float32)
rel_rec = torch.FloatTensor(rel_rec)
rel_send = torch.FloatTensor(rel_send)

if args.encoder == 'mlp':
    encoder = MLPEncoder(args.timesteps * args.dims, args.encoder_hidden,
                         args.edge_types,
                         args.encoder_dropout, args.factor)
elif args.encoder == 'cnn':
    encoder = CNNEncoder(args.dims, args.encoder_hidden,
                         args.edge_types,
                         args.encoder_dropout, args.factor)


elif args.encoder == 'mlpatten':
    encoder = MLPAttenEncoder(args.timesteps * args.dims, args.encoder_hidden,
                         args.edge_types,
                         args.encoder_dropout, args.factor)


if args.decoder == 'mlp':
    decoder = MLPDecoder(n_in_node=args.dims,
                         edge_types=args.edge_types,
                         msg_hid=args.decoder_hidden,
                         msg_out=args.decoder_hidden,
                         n_hid=args.decoder_hidden,
                         do_prob=args.decoder_dropout,
                         skip_first=args.skip_first)
elif args.decoder == 'rnn':
    decoder = RNNDecoder(n_in_node=args.dims,
                         edge_types=args.edge_types,
                         n_hid=args.decoder_hidden,
                         do_prob=args.decoder_dropout,
                         skip_first=args.skip_first)
elif args.decoder == 'sim':
    decoder = SimulationDecoder(
        loc_max, loc_min, vel_max, vel_min, args.suffix)

if args.load_folder:
    encoder_file = os.path.join(args.load_folder, 'encoder.pt')
    encoder.load_state_dict(torch.load(encoder_file))
    decoder_file = os.path.join(args.load_folder, 'decoder.pt')
    decoder.load_state_dict(torch.load(decoder_file))

    args.save_folder = False

optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()),
                       lr=args.lr)
scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_decay,
                                gamma=args.gamma)

# Linear indices of an upper triangular mx, used for acc calculation
triu_indices = get_triu_offdiag_indices(args.num_residues)
tril_indices = get_tril_offdiag_indices(args.num_residues)

if args.prior:
    prior = np.array([0.91, 0.03, 0.03, 0.03])  # TODO: hard coded for now
    print("Using prior")
    print(prior)
    log_prior = torch.FloatTensor(np.log(prior))
    log_prior = torch.unsqueeze(log_prior, 0)
    log_prior = torch.unsqueeze(log_prior, 0)
    log_prior = Variable(log_prior)

    if args.cuda:
        log_prior = log_prior.cuda()

if args.cuda:
    encoder.cuda()
    decoder.cuda()
    rel_rec = rel_rec.cuda()
    rel_send = rel_send.cuda()
    triu_indices = triu_indices.cuda()
    tril_indices = tril_indices.cuda()

rel_rec = Variable(rel_rec)
rel_send = Variable(rel_send)


def train(epoch, best_val_loss):
    t = time.time()
    nll_train = []
    acc_train = []
    kl_train = []
    mse_train = []
    edges_train = []
    probs_train = []

    encoder.train()
    decoder.train()

    for batch_idx, (data, relations) in enumerate(train_loader):

        if args.cuda:
            data, relations = data.cuda(), relations.cuda()
        data, relations = Variable(data), Variable(relations)
        optimizer.zero_grad()

        logits = encoder(data, rel_rec, rel_send)

        edges = gumbel_softmax(logits, tau=args.temp, hard=args.hard)
        prob = my_softmax(logits, -1)

        if args.decoder == 'rnn':
            output = decoder(data, edges, rel_rec, rel_send, 50,
                             burn_in=True,
                             burn_in_steps=args.timesteps - args.prediction_steps)
        else:
            output = decoder(data, edges, rel_rec, rel_send,
                             args.prediction_steps)

        target = data[:, :, 1:, :]

        loss_nll = nll_gaussian(output, target, args.var)

        if args.prior:
            loss_kl = kl_categorical(prob, log_prior, args.num_residues)
        else:
            loss_kl = kl_categorical_uniform(prob, args.num_residues,
                                             args.edge_types)

        loss = loss_nll + loss_kl

        acc = edge_accuracy(logits, relations)
        acc_train.append(acc)

        loss.backward()
        optimizer.step()

        mse_train.append(F.mse_loss(output, target).item())
        nll_train.append(loss_nll.item())
        kl_train.append(loss_kl.item())
        _, edges_t = edges.max(-1)
        edges_train.append(edges_t.data.cpu().numpy())
        probs_train.append(prob.data.cpu().numpy())
        #output.append(output.data.cpu().numpy())

    scheduler.step()
    nll_val = []
    acc_val = []
    kl_val = []
    mse_val = []

    encoder.eval()
    decoder.eval()
    for batch_idx, (data, relations) in enumerate(valid_loader):
        if args.cuda:
            data, relations = data.cuda(), relations.cuda()
        with torch.no_grad():

            logits = encoder(data, rel_rec, rel_send)
            edges = gumbel_softmax(logits, tau=args.temp, hard=True)
            prob = my_softmax(logits, -1)

            # validation output uses teacher forcing
            output = decoder(data, edges, rel_rec, rel_send, 1)

            target = data[:, :, 1:, :]
            loss_nll = nll_gaussian(output, target, args.var)
            loss_kl = kl_categorical_uniform(
                prob, args.num_residues, args.edge_types)

        acc = edge_accuracy(logits, relations)
        acc_val.append(acc)
        mse_val.append(F.mse_loss(output, target).item())
        nll_val.append(loss_nll.item())
        kl_val.append(loss_kl.item())

    print('Epoch: {:04d}'.format(epoch),
          'nll_train: {:.10f}'.format(np.mean(np.array(nll_train))),
          'kl_train: {:.10f}'.format(np.mean(np.array(kl_train))),
          'mse_train: {:.10f}'.format(np.mean(np.array(mse_train))),
          'acc_train: {:.10f}'.format(np.mean(np.array(acc_train))),
          'nll_val: {:.10f}'.format(np.mean(np.array(nll_val))),
          'kl_val: {:.10f}'.format(np.mean(np.array(kl_val))),
          'mse_val: {:.10f}'.format(np.mean(np.array(mse_val))),
          'acc_val: {:.10f}'.format(np.mean(np.array(acc_val))),
          'time: {:.4f}s'.format(time.time() - t))
    edges_train = np.concatenate(edges_train)
    probs_train = np.concatenate(probs_train)

    # 将所有结果写入文件
    training_log = open('logs/training_log.txt', 'w')
    print('Epoch: {:04d}'.format(epoch),
      'nll_train: {:.10f}'.format(np.mean(np.array(nll_train))),
      'kl_train: {:.10f}'.format(np.mean(np.array(kl_train))),
      'mse_train: {:.10f}'.format(np.mean(np.array(mse_train))),
      'acc_train: {:.10f}'.format(np.mean(np.array(acc_train))),
      'nll_val: {:.10f}'.format(np.mean(np.array(nll_val))),
      'kl_val: {:.10f}'.format(np.mean(np.array(kl_val))),
      'mse_val: {:.10f}'.format(np.mean(np.array(mse_val))),
      'acc_val: {:.10f}'.format(np.mean(np.array(acc_val))),
      'time: {:.4f}s'.format(time.time() - t), file=training_log)
    training_log.flush()

    if args.save_folder and np.mean(np.array(nll_val)) < best_val_loss:
        torch.save(encoder.state_dict(), encoder_file)
        torch.save(decoder.state_dict(), decoder_file)
        print('Best model so far, saving...')
        print('Epoch: {:04d}'.format(epoch),
              'nll_train: {:.10f}'.format(np.mean(np.array(nll_train))),
              'kl_train: {:.10f}'.format(np.mean(np.array(kl_train))),
              'mse_train: {:.10f}'.format(np.mean(np.array(mse_train))),
              'acc_train: {:.10f}'.format(np.mean(np.array(acc_train))),
              'nll_val: {:.10f}'.format(np.mean(np.array(nll_val))),
              'kl_val: {:.10f}'.format(np.mean(np.array(kl_val))),
              'mse_val: {:.10f}'.format(np.mean(np.array(mse_val))),
              'acc_val: {:.10f}'.format(np.mean(np.array(acc_val))),
              'time: {:.4f}s'.format(time.time() - t), file=log)
        log.flush()

    # return encoder, decoder, edges_train, probs_train, np.mean(np.array(nll_val))
    return encoder, decoder, edges_train, probs_train, output, np.mean(np.array(nll_val)), np.mean(np.array(acc_val))

def test():
    acc_test = []
    nll_test = []
    kl_test = []
    mse_test = []
    edges_test = []
    probs_test = []
    tot_mse = 0
    counter = 0

    encoder.eval()
    decoder.eval()
    encoder.load_state_dict(torch.load(encoder_file))
    decoder.load_state_dict(torch.load(decoder_file))

    for batch_idx, (data, relations) in enumerate(test_loader):
        if args.cuda:
            data, relations = data.cuda(), relations.cuda()

        with torch.no_grad():
            #        assert (data.size(2) - args.timesteps) >= args.timesteps
            assert (data.size(2)) >= args.timesteps

            data_encoder = data[:, :, :args.timesteps, :].contiguous()
            data_decoder = data[:, :, -args.timesteps:, :].contiguous()

            logits = encoder(data_encoder, rel_rec, rel_send)
            edges = gumbel_softmax(logits, tau=args.temp, hard=True)

            prob = my_softmax(logits, -1)

            output = decoder(data_decoder, edges, rel_rec, rel_send, 1)

            target = data_decoder[:, :, 1:, :]
            loss_nll = nll_gaussian(output, target, args.var)
            loss_kl = kl_categorical_uniform(
                prob, args.num_residues, args.edge_types)

            acc = edge_accuracy(logits, relations)

            acc_test.append(acc)
            mse_test.append(F.mse_loss(output, target).item())
            nll_test.append(loss_nll.item())
            kl_test.append(loss_kl.item())
            _, edges_t = edges.max(-1)
            edges_test.append(edges_t.data.cpu().numpy())
            probs_test.append(prob.data.cpu().numpy())

            # For plotting purposes
            if args.decoder == 'rnn':
                if args.dynamic_graph:
                    output = decoder(data, edges, rel_rec, rel_send, 50,
                                     burn_in=False, burn_in_steps=args.timesteps,
                                     dynamic_graph=True, encoder=encoder,
                                     temp=args.temp)
                else:
                    output = decoder(data, edges, rel_rec, rel_send, 50,
                                     burn_in=True, burn_in_steps=args.timesteps)

                target = data[:, :, 1:, :]

            else:
                data_plot = data[:, :, 0:0 + 21,
                                 :].contiguous()
                output = decoder(data_plot, edges, rel_rec, rel_send, 20)
                target = data_plot[:, :, 1:, :]

            mse = ((target - output) ** 2).mean(dim=0).mean(dim=0).mean(dim=-1)
            tot_mse += mse.data.cpu().numpy()
            counter += 1

    mean_mse = tot_mse / counter
    mse_str = '['
    for mse_step in mean_mse[:-1]:
        mse_str += " {:.12f} ,".format(mse_step)
    mse_str += " {:.12f} ".format(mean_mse[-1])
    mse_str += ']'

    print('--------------------------------')
    print('--------Testing-----------------')
    print('--------------------------------')
    print('nll_test: {:.10f}'.format(np.mean(nll_test)),
          'kl_test: {:.10f}'.format(np.mean(kl_test)),
          'mse_test: {:.10f}'.format(np.mean(mse_test)),
          'acc_test: {:.10f}'.format(np.mean(acc_test)))
    print('MSE: {}'.format(mse_str))
    edges_test = np.concatenate(edges_test)
    probs_test = np.concatenate(probs_test)

    if args.save_folder:
        print('--------------------------------', file=log)
        print('--------Testing-----------------', file=log)
        print('--------------------------------', file=log)
        print('nll_test: {:.10f}'.format(np.mean(nll_test)),
              'kl_test: {:.10f}'.format(np.mean(kl_test)),
              'mse_test: {:.10f}'.format(np.mean(mse_test)),
              'acc_test: {:.10f}'.format(np.mean(acc_test)),
              file=log)
        print('MSE: {}'.format(mse_str), file=log)
        log.flush()
    return edges_test, probs_test


# Train model
print("Start Training...")
t_total = time.time()
best_val_loss = np.inf
best_epoch = 0
x_data = np.arange(1, args.epochs + 1)
y_data = []
z_data = []
for epoch in range(args.epochs):
    encoder, decoder, edges_train, probs_train, pred, val_loss, val_acc = train(
        epoch, best_val_loss)
    # encoder, decoder, edges_train, probs_train, val_loss = train(
    #     epoch, best_val_loss)
    y_data.append(val_loss)
    z_data.append(val_acc)
    #print(pred.shape)
    #print(probs_train.shape)
    # print('Epoch '+str(epoch)+' with val loss:'+str(val_loss))
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch

np.save(str(args.save_folder)+'/val_loss.npy', y_data)
np.save(str(args.save_folder)+'/val_acc.npy', z_data)
np.save(str(args.save_folder)+'/pred_train.npy', pred.cpu().numpy())
np.save(str(args.save_folder)+'/out_edges_train.npy', edges_train)
np.save(str(args.save_folder)+'/out_probs_train.npy', probs_train)
print("Optimization Finished!")
print("Best Epoch: {:04d}".format(best_epoch))
if args.save_folder:
    print("Best Epoch: {:04d}".format(best_epoch), file=log)
    log.flush()


# Test
edges_test, probs_test = test()
print(edges_test.shape, probs_test.shape)
if log is not None:
    print(save_folder)
    log.close()
