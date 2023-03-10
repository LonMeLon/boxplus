import time
#import wandb
import torch
from torch.utils.data import DataLoader
import math
import argparse
import os
import json
from utils import *
from dataset import *
from softbox import SoftBox
from gumbel_box import GumbelBox
from gumbel_dist import GumbelBoxDist

box_model = {'softbox': SoftBox,
             'gumbel': GumbelBox,
			 'gumbeldist' : GumbelBoxDist}

def random_negative_sampling(samples, probs, vocab_size, ratio, max_num_neg_sample):
	with torch.no_grad():
		negative_samples = samples.repeat(ratio, 1)[:max_num_neg_sample, :]
		neg_sample_size = negative_samples.size()[0]
		x = torch.arange(neg_sample_size)
		y = torch.randint(negative_samples.size()[1], (neg_sample_size,))
		negative_samples[x, y] = torch.randint(vocab_size, (neg_sample_size,))
		negative_probs = torch.zeros(negative_samples.size()[0], dtype=torch.long)
		samples_aug = torch.cat([samples, negative_samples], dim=0)
		probs_aug = torch.cat([probs, negative_probs], dim=0)
	return samples_aug, probs_aug

def train_func(train_data, vocab_size, random_negative_sampling_ratio, thres, optimizer, criterion, device, batch_size, model):
	pos_batch_size = math.ceil(batch_size/(random_negative_sampling_ratio+1))
	max_neg_batch_size = batch_size - pos_batch_size

	# Train the model
	train_loss = [0, 0, 0]
	train_acc = 0.
	train_size = 0.
	data = DataLoader(train_data, batch_size=pos_batch_size, shuffle=True)
	model.train()
	for ids, cls in data:
		optimizer.zero_grad()
		ids_aug, cls_aug = random_negative_sampling(ids, cls, vocab_size, random_negative_sampling_ratio, max_neg_batch_size)
		ids_aug, cls_aug = ids_aug.to(device), cls_aug.to(device)
		# loss
		'''
		output = model(ids_aug)
		loss = criterion(output, cls_aug)
		loss_regular = model.all_boxes_volumes().mean()
		loss = loss + 0.01 * loss_regular
		'''
		Index_overlap, Index_disjoint, output = model(ids_aug)
		if Index_overlap[0].shape[0] > 0:
			loss_overlap = criterion(output[Index_overlap], cls_aug[Index_overlap])
		else:
			loss_overlap = torch.zeros(1).to('cuda:0')
		if Index_disjoint[0].shape[0] > 0:
			loss_disjoint = (cls_aug[Index_disjoint] * output[Index_disjoint][:, 1]).mean()
			#print(Index_disjoint, loss_disjoint)
		else:
			loss_disjoint = torch.zeros(1).to('cuda:0')
		loss_regular = model.all_boxes_volumes()
		loss = loss_overlap + loss_disjoint + loss_regular
		#print(loss_overlap, loss_disjoint, 0.01 * loss_regular)
		
		# other
		train_loss[0] += loss_overlap.item()
		train_loss[1] += loss_regular.item()
		train_loss[2] += loss_disjoint.item()
		loss.backward()
		optimizer.step()
		train_acc += ((output[Index_overlap][:, 1] >= thres).long() == cls_aug[Index_overlap]).sum().item()
		train_size += ids_aug.size()[0]

	return [lo / train_size for lo in train_loss], train_acc / train_size

def test(test_data, thres, criterion, device, batch_size, model):
	test_loss = [0, 0, 0]
	acc = 0
	scores= []
	true = 0
	data = DataLoader(test_data, batch_size=batch_size)
	model.eval()
	for ids, cls in data:
		ids, cls = ids.to(device), cls.to(device)
		with torch.no_grad():
			'''
			output = model(ids)
			loss = criterion(output, cls)
			loss += loss.item()
			acc += ((output[:, 1] >= thres).long() == cls).sum().item()
			scores.extend(output[:, 0])
			true += cls.sum()
			'''
			Index_overlap, Index_disjoint, output = model(ids)
			if Index_overlap[0].shape[0] > 0:
				loss_overlap = criterion(output[Index_overlap], cls[Index_overlap])
			else:
				loss_overlap = torch.zeros(1).to('cuda:0')
			if Index_disjoint[0].shape[0] > 0:
				loss_disjoint = (cls[Index_disjoint] * output[Index_disjoint][:, 1]).mean()
				#print(Index_disjoint)
			else:
				loss_disjoint = torch.zeros(1).to('cuda:0')
			loss_regular = model.all_boxes_volumes()
			loss = loss_overlap + loss_disjoint + loss_regular 
			#test_loss += loss.item()
			test_loss[0] += loss_overlap.item()
			test_loss[1] += loss_regular.item()
			test_loss[2] += loss_disjoint.item()

			acc += ((output[Index_overlap][:, 1] >= thres).long() == cls[Index_overlap]).sum().item()
			#scores.extend(output[:, 0])
			#true += cls.sum()
			

	return [lo / len(test_data) for lo in test_loss], acc / len(test_data)

def resume_and_test(resume_from, test_data_path, vocab_path):
	ckpt = torch.load(resume_from)
	args = ckpt['args']

	test_dataset = PairDataset(test_data_path)
	word2idx = get_vocab(vocab_path)

	VOCAB_SIZE = len(word2idx)
	NUN_CLASS = 2

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	#model = box_model[args.model](VOCAB_SIZE, args.box_embedding_dim, NUN_CLASS, [1e-4, 0.2], [-0.1, 0], args).to(device)
	#model = box_model[args.model](VOCAB_SIZE, args.box_embedding_dim, NUN_CLASS, [1e-4, 0.2], [-0.1, 0], args).to(device)
	model.load_state_dict(ckpt['state_dict'], strict=True)

	criterion = torch.nn.CrossEntropyLoss().to(device)

	loss, acc = test(test_dataset, args.prediction_thres, criterion, device, args.batch_size, model)
	print(f'Test Loss: {loss:.8f}\t|\tTest Acc: {acc * 100:.2f}%')

def main(args):
	#wandb.init(project="basic_box", config=args)
	startime = time.time()
	train_dataset = PairDataset(args.train_data_path)
	test_dataset = PairDataset(args.test_data_path)
	word2idx = get_vocab(args.vocab_path)
	print((time.time() - startime))

	VOCAB_SIZE = len(word2idx)
	NUN_CLASS = 2

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	#model = box_model[args.model](VOCAB_SIZE, args.box_embedding_dim, NUN_CLASS, [1e-4, 0.2], [-0.1, 0], args).to(device)
	#model = box_model[args.model](VOCAB_SIZE, args.box_embedding_dim, NUN_CLASS, [1e-3, 3], [-0.1, 0], args).to(device)
	model = box_model[args.model](VOCAB_SIZE, args.box_embedding_dim, NUN_CLASS, [1e-4, 1.2], [-0.1, 0], args).to(device)

	criterion = torch.nn.CrossEntropyLoss().to(device)
	optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

	#wandb.watch(model)

	for epoch in range(args.epochs):

		start_time = time.time()
		train_loss, train_acc = train_func(train_dataset, VOCAB_SIZE, args.random_negative_sampling_ratio, args.prediction_thres,
										   optimizer, criterion, device, args.batch_size, model)
		valid_loss, valid_acc = test(test_dataset, args.prediction_thres, criterion, device, args.batch_size, model)

		#wandb.log({'train loss': train_loss, 'train accuracy': train_acc, 'valid loss': valid_loss, 'valid accuracy': valid_acc})

		secs = int(time.time() - start_time)
		mins = secs / 60
		secs = secs % 60

		print('Epoch: %d' %(epoch + 1), " | time in %d minutes, %d seconds" %(mins, secs))
		print(f'\tLoss: {train_loss[0]:.8f}(train)\tLoss: {train_loss[1]:.8f}\tLoss: {train_loss[2]:.8f}|\tAcc: {train_acc * 100:.2f}%(train)')
		print(f'\tLoss: {valid_loss[0]:.8f}(valid)\tLoss: {train_loss[1]:.8f}\tLoss: {train_loss[2]:.8f}|\tAcc: {valid_acc * 100:.2f}%(valid)')

		# save model if the valid_acc is the current best or better than 99.8%
		best_valid_acc = 0.
		used_id = -1
		if not os.path.exists("checkpoints"): os.mkdir("checkpoints")
		if not os.path.exists(args.save_to): os.mkdir(args.save_to)
		history_file = os.path.join(args.save_to, "history.json")
		if os.path.exists(history_file):
			with open(history_file, "r") as f:
				history = json.loads(f.read())
				best_valid_acc = history["best_valid_acc"]
				used_id = history["used_id"]
		if valid_acc > best_valid_acc:
			best_valid_acc = valid_acc
			torch.save({'args': args,
						'epoch': epoch,
						'state_dict': model.state_dict(),
						'optimizer': optimizer.state_dict()},
					   os.path.join(args.save_to, 'best_checkpoint.pth'))
		# if valid_acc >= 0.9995:
		# 	used_id += 1
		# 	torch.save({'args': args,
		# 				'epoch': epoch,
		# 				'state_dict': model.state_dict(),
		# 				'optimizer': optimizer.state_dict()},
		# 			   os.path.join(args.save_to, 'checkpoint_%d.pth' % used_id))
		with open(history_file, "w") as f:
			f.write(json.dumps({"best_valid_acc": best_valid_acc, "used_id": used_id}))


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	# data parameters
	parser.add_argument('--train_data_path', type=str, default='data/full_wordnet_noneg.tsv', help='path to train data')
	parser.add_argument('--test_data_path', type=str, default='data/full_wordnet.tsv', help='path to test data')
	parser.add_argument('--vocab_path', type=str, default='data/full_wordnet_vocab.tsv', help='path to vocab')
	# training parameters
	parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA training (eg. no nvidia GPU)')
	parser.add_argument('--random_negative_sampling_ratio', type=int, default=1, help='sample this many random negatives for each positive.')
	parser.add_argument('--batch_size', type=int, default=800, help='batch size for training will be 2**LOG_BATCH_SIZE')
	parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate')
	parser.add_argument('--epochs', type=int, default=80, help='number of epochs to train')
	parser.add_argument('--prediction_thres', type=float, default=0.5, help='the probability threshold for prediction')
	# model parameters
	parser.add_argument('--model', type=str, default='softbox', help='model type: choose from softbox, gumbel')
	parser.add_argument('--box_embedding_dim', type=int, default=50, help='box embedding dimension')
	parser.add_argument('--softplus_temp', type=float, default=1.0, help='beta of softplus function')
	# gumbel box parameters
	parser.add_argument('--gumbel_beta', type=float, default=0.01, help='beta value for gumbel distribution')
	parser.add_argument('--scale', type=float, default=1.0, help='scale value for gumbel distribution')
	# a parameter can be set to a model checkpoint path or left as None
	# If set, the checkpoint will be resumed and tested; the user needs to specify test_data_path and vocab_path but not others
	# Other parameters will be restored from the model checkpoint
	parser.add_argument('--resume_and_test', type=str, default=None, help='path to a model checkpoint to be resumed and tested')

	args = parser.parse_args()
	args.save_to = "./checkpoints/" + args.model

	if args.resume_and_test is not None:
		resume_and_test(args.resume_and_test, args.test_data_path, args.vocab_path)
	else:
		main(args)


