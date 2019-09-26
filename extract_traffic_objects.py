#!/usr/bin/python3

import os, sys
import numpy as np
from scipy.sparse import csr_matrix
import pickle
import socket

import scapy, csv
from scapy.all import *
from collections import Counter

import datetime, time
from datetime import datetime

import logging
from logging.handlers import RotatingFileHandler

import time

import itertools
from copy import copy

from contextlib import redirect_stdout
from io import StringIO

import warnings
warnings.simplefilter("error")

# TCP status code
FIN = 0x01
SYN = 0x02
RST = 0x04
PSH = 0x08
ACK = 0x10
URG = 0x20
ECE = 0x40
CWR = 0x80

STATUS_CODES = sorted('FSRPAUEC')
STATUS_COMBINATIONS = ['']
for i in range(len(STATUS_CODES)):
	STATUS_COMBINATIONS += [ ''.join(c) for c in itertools.combinations(STATUS_CODES, i+1) ]

PROTOS = [ name[8:].lower() for name,_ in vars(socket).items() if name.startswith('IPPROTO') ]
PROTOS = sorted(PROTOS)

ICMP_CODES = [ i for i in range(256) ]

# Timeouts per protocol
UDP_TIMEOUT = 60 # timeout in seconds: TIE use 1 m
TCP_TIMEOUT = 300 # timeout in seconds: TIE use 1 m
OTHER_TIMEOUT = 60 # timeout in seconds: TIE use 1 m

TIME_WINDOW = 1e-3 # timw window in s

MAX_PKT_PER_TO = 32

proto_names = {num:name[8:] for name,num in vars(socket).items() if name.startswith('IPPROTO')}

class LabeledPacket(object):
	'''
	This class provide a wrapepr for packets to associate them a label.
	'''
	def __init__(self, packet, label):

		self.packet = packet
		self.label = label

class CustomTrafficObjectPickleException(Exception):
	def __init__(self, error_message):
		self.error_message = error_message
	def __str__(self):
		return str(self.error_message)

def expand(x):
	yield x.name
	while x.payload:
		x = x.payload
		yield x.name

class TrafficObjectPickle(object):
	
	def __init__(self, label_file, pcap_file, out_pickle, dataset_name, traffic_object_name):
		'''
		MatrixExtraction class constructor
		:param self:
		:param label_file: traffic trace in .pickle format
		:param pcap_file: timestamp of the pcap file
		:param out_pickle: pickle file serializing lopez2017network DL-input extracted from pcap_file
		'''
		if os.path.isfile(label_file):
			self.label_file = label_file
		else:
			self.debug_log.debug('Error: labels file does not exist')
			raise CustomTrafficObjectPickleException('Error: labels file does not exist')
		if os.path.isfile(pcap_file):
			pcap_reader = PcapReader(pcap_file)
			self.pcap_packets = [ p for p in pcap_reader ]
			pcap_reader.close()
		else:
			self.debug_log.debug('Error: pcap file does not exist')
			raise CustomTrafficObjectPickleException('Error: pcap file does not exist')
		self.out_pickle = out_pickle
		self.dataset_name = dataset_name
		self.traffic_object_name = traffic_object_name

		self.labels_dict = {}
		self.real_traffic_objects = {}
		self.dataset_per_traffic_object = {}
		self.dataset_per_packet = {}

		self.packet_samples = []
		self.packet_labels = []
		self.traffic_object_samples = []
		self.traffic_object_labels = []
		
		self.debug_log = self.setup_logger('debug', 'extract_traffic_object.log', logging.DEBUG)
		time_base_dir = './time_base/'
		if not os.path.exists(time_base_dir):
			os.mkdir(time_base_dir)
		self.csv_time_base_file_name = time_base_dir+self.label_file.split('/')[-1].split('_')[0]+'.csv.pickle'
		self.pcap_time_base_file_name = time_base_dir+pcap_file.split('/')[-1].split('_')[0]+'.pcap.pickle'

	def setup_logger(self, logger_name, log_file, level=logging.INFO):
		'''
		Return a log object associated to <log_file> with specific formatting and rotate handling.
		'''

		l = logging.getLogger(logger_name)

		if not getattr(l, 'handler_set', None):
			logging.Formatter.converter = time.gmtime
			formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s')

			rotatehandler = RotatingFileHandler(log_file, mode='a', maxBytes=10485760, backupCount=30)
			rotatehandler.setFormatter(formatter)

			l.addHandler(rotatehandler)
			l.setLevel(level)
			l.handler_set = True
		
		elif not os.path.exists(log_file):
			logging.Formatter.converter = time.gmtime
			formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s')

			rotatehandler = RotatingFileHandler(log_file, mode='a', maxBytes=10485760, backupCount=30)
			rotatehandler.setFormatter(formatter)

			l.addHandler(rotatehandler)
			l.setLevel(level)
			l.handler_set = True
		
		return l

	def factory_methods(self, discriminator, name, argv=[]):
		if discriminator == 'd':
			factory_method = getattr(self, '%s_%s' % (self.dataset_name, name))
		if discriminator == 't':
			factory_method = getattr(self, '%s_%s' % (self.traffic_object_name, name))
		return factory_method(*argv)

	def UNSWIOT_retrieve_labels(self):
		'''
		Deserialize pickle dictionary with correspondance mac - labels
		'''
		with open(self.label_file, 'rb') as label_file:
			self.labels_dict =  pickle.load(label_file)

	def BOTIOT_retrieve_labels(self):

		self.labels_dict = {}
		try:
			with open(self.label_file+'.pickle','rb') as label_file:
				self.labels_dict = pickle.load(label_file)
		except:
			with open(self.label_file, 'r') as label_file:
				attributes = label_file.readline().replace('"','').strip().split(';')
				rows = label_file.readlines()
				for row in rows:
					row = row.replace('"','').strip().split(';')
					proto = row[attributes.index('proto')]
					if proto in ['udp','tcp']:
						key = ('%s,%s,%s,%s,%s' % (row[attributes.index('saddr')], row[attributes.index('sport')], row[attributes.index('daddr')], row[attributes.index('dport')], row[attributes.index('proto')]))
						label = [ row[attributes.index('attack')], row[attributes.index('category')], row[attributes.index('subcategory')] ]
						validity_range = [ float(row[attributes.index('stime')]), float(row[attributes.index('ltime')])]
					else:
						key = ('%s,%s,%s,%s,%s' % (row[attributes.index('saddr')], 0, row[attributes.index('daddr')], 0, row[attributes.index('proto')]))
						label = [ row[attributes.index('attack')], row[attributes.index('category')], row[attributes.index('subcategory')] ]
						validity_range = [ float(row[attributes.index('stime')]), float(row[attributes.index('ltime')])]
					if key not in self.labels_dict:
						self.labels_dict[key] = []
					self.labels_dict[key].append([ validity_range, label ])
			with open(self.label_file+'.pickle','wb') as label_file:
				pickle.dump(self.labels_dict, label_file, pickle.HIGHEST_PROTOCOL)

	def CIC18_retrieve_labels(self):
		
		self.labels_dict = {}
		try:
			with open(self.label_file+'.pickle','rb') as label_file:
				self.labels_dict = pickle.load(label_file)
		except:
			with open(self.label_file, 'r') as label_file:
				attributes = label_file.readline().strip().split(',')
				rows = label_file.readlines()
				for row in rows:
					row = row.strip().split(',')
					key = ('%s,%s' % (row[attributes.index('threat_IP')], row[attributes.index('victim_IP')]))
					label = [ row[attributes.index('label_attack')], row[attributes.index('label_vector')], row[attributes.index('label_type')] ]
					start_time, finish_time = ':'.join(row[attributes.index('start_time')].split(':')[:2]), ':'.join(row[attributes.index('finish_time')].split(':')[:2])
					current_date = '-'.join(self.label_file.split('.')[0].split('-')[1:])
					date_format = '%d-%m-%Y %H:%M:%S.%f'
					validity_range = [
						time.mktime(time.strptime('%s %s:00.000000'%(current_date, start_time), date_format)),
						time.mktime(time.strptime('%s %s:59.999999'%(current_date, finish_time), date_format))
						]
					if key not in self.labels_dict:
						self.labels_dict[key] = []
					self.labels_dict[key].append([ validity_range, label ])
			with open(self.label_file+'.pickle','wb') as label_file:
				pickle.dump(self.labels_dict, label_file, pickle.HIGHEST_PROTOCOL)

	# def compute_septuple(self, septuple, old_divider, new_divider, new_positions):
	# 	'''
	# 	:param self:
	# 	:param septuple: string septuple to transform, contains src ip and port, dst ip and port, and protocol in a user known positions, i.e. '192.168.200.1-23435-192.168.200.2-32412-17'
	# 	:param old_divider: divider of elements above mentioned, i.e. '-'
	# 	:param new_divider: new divider user wants to use, i.e. ','
	# 	:param new_positions: new indexing list of elements' septuple, i.e. [0,2,1,3,4]
	# 	:return computed_septuple: string septuple transformed with new positions and nwe divider, i.e. '192.168.200.1,192.168.200.2,23435,32412,17'
	# 	'''
	# 	computed_septuple = new_divider.join([septuple.split(old_divider)[i] for i in new_positions])
	# 	return computed_septuple

	def get_packet_layers(self,packet):
		counter = 0
		while True:
			layer = packet.getlayer(counter)
			if layer is None:
				break
			yield layer
			counter += 1

	def get_septuple(self,p):
		dl_src = '0'
		dl_dst = '0'
		n_src = '0'
		n_dst = '0'
		t_src = '0'
		t_dst = '0'
		proto = '0'

		# if p.haslayer(ICMP):
		# 	print('/'.join([layer.name for layer in self.get_packet_layers(p)]))

		try:
			# If datalink layer is Ethernet, we save MAC addresses and next inspect inner protocol
			if p.haslayer(Ether):
				dl_src = p[Ether].src
				dl_dst = p[Ether].dst
				# If network layer is IP, we save IP addresses and new inspect inner protocol
				if p.haslayer(IP):
					n_src = p[IP].src
					n_dst = p[IP].dst
					# If transport layer is TCP, we save TPC addresses and set protocol to tcp
					if p.haslayer(TCP):
						t_src = str(p[TCP].sport)
						t_dst = str(p[TCP].dport)
						proto = 'tcp'
					# If transport layer is UDP, we save UDP addresses and set protocol to udp
					elif p.haslayer(UDP) and not p.haslayer(ICMP):
						t_src = str(p[UDP].sport)
						t_dst = str(p[UDP].dport)
						proto = 'udp'
					# If IP contains an ICMP header, we set protocol to icmp
					elif p.haslayer(ICMP):
						proto = 'icmp'
				elif p.haslayer(IPv6):
					n_src = p[IPv6].src
					n_dst = p[IPv6].dst
					if p.haslayer(IPv6ExtHdrHopByHop):
						proto = proto_names[p[IPv6ExtHdrHopByHop].nh].lower()
					elif p.haslayer(UDP):
						t_src = str(p[UDP].sport)
						t_dst = str(p[UDP].dport)
						proto = 'udp'
					else:
						proto = proto_names[p[IPv6].nh].lower()
				elif p.haslayer(ARP):
					n_src = p[ARP].psrc
					n_dst = p[ARP].pdst
					proto = 'arp'
			# 	elif p.haslayer(EAPOL):
			# 		proto = 'eapol'
			# if p.haslayer(Dot3):
			# 	dl_src = p[Dot3].src
			# 	dl_dst = p[Dot3].dst
			# 	if p.haslayer(LLC):
			# 		proto = 'llc'
		except Exception as e:
			pass
			# print(p)
			# print(p[IPv6].nh)
			# input()

		# if p.haslayer(ICMP):
		# 	print(dl_src,dl_dst,n_src,n_dst,t_src,t_dst,proto)

		return dl_src,dl_dst,n_src,n_dst,t_src,t_dst,proto

	# TODO: provide an external initialization of TIMEOUTS
	def CONNECTION_pcap_extraction(self):
		stat = {}
		T_seqs = {}
		for packet in self.pcap_packets:
			try:
				dl_src, dl_dst, n_src, n_dst, t_src, t_dst, proto = self.get_septuple(packet)
				src_socket = '%s,%s,%s' % (dl_src, n_src, t_src)
				dst_socket = '%s,%s,%s' % (dl_dst, n_dst, t_dst)
				septuple = '%s,%s,%s' % (src_socket, dst_socket, proto)
				inv_septuple = '%s,%s,%s' % (dst_socket, src_socket, proto)
				current_septuple = septuple
				if septuple not in self.real_traffic_objects and inv_septuple not in self.real_traffic_objects:
					self.real_traffic_objects[septuple] = []
					stat[septuple] = 'SO'
					T_seqs[septuple] = []
				elif septuple not in self.real_traffic_objects:
					current_septuple = inv_septuple
				T = packet.time
				time_diff = 0.0
				try:
					time_diff = T - T_seqs[current_septuple][-1][-1]
				except:
					pass
				# Status of SFM:
				#  APPEND_PACKET: P
				#  APPEND_SEGMENT: S
				#  OPEN: O
				#  HALF_CLOSED: H
				#  CLOSED: C
				if proto == 'tcp':
					flags = packet[TCP].flags
					flags = flags & ( FIN | SYN | RST | PSH | ACK | URG | ECE | CWR )
					if len(self.real_traffic_objects[current_septuple]) == 0 or flags == 'S' or time_diff > TCP_TIMEOUT or ('C' in stat[current_septuple] and 'A' not in flags):
						stat[current_septuple] = 'SO'
					elif 'R' in flags or ('F' in flags and 'H' in stat[current_septuple]):
						stat[current_septuple] = 'PC'
					elif 'F' in flags and 'O' in stat[current_septuple]:
						stat[current_septuple] = 'PH'
				elif proto == 'udp':
					flags = ''
					if len(self.real_traffic_objects[current_septuple]) == 0 or time_diff > UDP_TIMEOUT:
						stat[current_septuple] = 'SO'
					else:
						stat[current_septuple] = 'PO'
				else:
					flags = ''
					if len(self.real_traffic_objects[current_septuple]) == 0 or time_diff > OTHER_TIMEOUT:
						stat[current_septuple] = 'SO'
					else:
						stat[current_septuple] = 'PO'
				current_stat = stat[current_septuple]
				if 'S' in current_stat:
					T_seqs[current_septuple].append([])
					self.real_traffic_objects[current_septuple].append([])
					stat[current_septuple] = 'P'+stat[current_septuple][1]
				T_seqs[current_septuple][-1].append(T)
				self.real_traffic_objects[current_septuple][-1].append(packet)
			except:
				traceback.print_exc(file=sys.stderr)
				self.debug_log.debug('Packet extraction failed: %s\n' % septuple)
				continue

	def BICHANNEL_pcap_extraction(self):
		for packet in self.pcap_packets:
			try:
				dl_src, dl_dst, n_src, n_dst, t_src, t_dst, proto = self.get_septuple(packet)
				src_channel = '%s,%s' % (dl_src,n_src)
				dst_channel = '%s,%s' % (dl_dst,n_dst)
				quadruple = '%s,%s' % (src_channel,dst_channel)
				inv_quadruple = '%s,%s' % (dst_channel,src_channel)
				current_quadruple = quadruple
				if quadruple not in self.real_traffic_objects and inv_quadruple not in self.real_traffic_objects:
					self.real_traffic_objects[quadruple] = []
				elif quadruple not in self.real_traffic_objects:
					current_quadruple = inv_quadruple
				self.real_traffic_objects[current_quadruple].append(packet)
			except:
				traceback.print_exc(file=sys.stderr)
				self.debug_log.debug('Packet extraction failed: %s\n' % quadruple)
				continue

	def BIFLOW_pcap_extraction(self):
		import time
		import progressbar

		for _,packet in zip(progressbar.progressbar(range(len(self.pcap_packets)), redirect_stdout=True),self.pcap_packets):
		# for packet in self.pcap_packets:
			try:
				dl_src, dl_dst, n_src, n_dst, t_src, t_dst, proto = self.get_septuple(packet)
				src_socket = '%s,%s,%s' % (dl_src,n_src,t_src)
				dst_socket = '%s,%s,%s' % (dl_dst,n_dst,t_dst)
				septuple = '%s,%s,%s' % (src_socket,dst_socket,proto)
				inv_septuple = '%s,%s,%s' % (dst_socket,src_socket,proto)
				# print(septuple)
				# input()
				current_septuple = septuple
				if septuple not in self.real_traffic_objects and inv_septuple not in self.real_traffic_objects:
					self.real_traffic_objects[septuple] = []
				elif septuple not in self.real_traffic_objects:
					current_septuple = inv_septuple
				if len(self.real_traffic_objects[current_septuple]) < MAX_PKT_PER_TO:
					self.real_traffic_objects[current_septuple].append(packet)
			except:
				traceback.print_exc(file=sys.stderr)
				self.debug_log.debug('Packet extraction failed: %s\n' % septuple)
				continue

		keys = self.real_traffic_objects.keys()
		print(len([key for key in keys if 'icmp' in key]))
		print(np.unique([len(self.real_traffic_objects[key]) for key in keys if 'icmp' in key], return_counts=True))

	def BAG_OF_SENT_PACKETS_pcap_extraction(self):
		import time
		import progressbar

		times = {}

		# for i,packet in zip(progressbar.progressbar(range(len(self.pcap_packets)), redirect_stdout=True),self.pcap_packets):
		for packet in self.pcap_packets:
			try:
				dl_src, dl_dst, n_src, n_dst, t_src, t_dst, proto = self.get_septuple(packet)
				if(dl_src==''):
					continue
				src_mac = dl_src
				if src_mac not in self.real_traffic_objects:
					self.real_traffic_objects[src_mac] = []
				if src_mac in times:
					if packet.time - times[src_mac] > TIME_WINDOW:
						times[src_mac] = times[src_mac] + TIME_WINDOW
						self.real_traffic_objects[src_mac].append([])
				else:
					times[src_mac] = packet.time
					self.real_traffic_objects[src_mac].append([])
				self.real_traffic_objects[src_mac][-1].append(packet)
			except:
				traceback.print_exc(file=sys.stderr)
				self.debug_log.debug('Packet extraction failed: %s\n' % src_mac)
				continue

	def BAG_OF_RECEIVED_PACKETS_pcap_extraction(self):
		import time
		import progressbar

		times = {}

		# for i,packet in zip(progressbar.progressbar(range(len(self.pcap_packets)), redirect_stdout=True),self.pcap_packets):
		for packet in self.pcap_packets:
			try:
				dl_src, dl_dst, n_src, n_dst, t_src, t_dst, proto = self.get_septuple(packet)
				dst_ip = n_dst
				if dst_ip not in self.real_traffic_objects:
					self.real_traffic_objects[dst_ip] = []
				if dst_ip in times:
					if packet.time - times[dst_ip] > TIME_WINDOW:
						times[dst_ip] = packet.time
						self.real_traffic_objects[dst_ip].append([])
				else:
					times[dst_ip] = packet.time
					self.real_traffic_objects[dst_ip].append([])
				self.real_traffic_objects[dst_ip][-1].append(packet)
			except:
				traceback.print_exc(file=sys.stderr)
				self.debug_log.debug('Packet extraction failed: %s\n' % dst_ip)
				continue

	def check_chksum(self, proto, packet):
		if proto in packet:
			temp_packet = copy(packet)
			old_chksum = temp_packet[proto].chksum
			del temp_packet[proto].chksum
			temp_stdout = StringIO()
			with redirect_stdout(temp_stdout):
				temp_packet.show2()
			reconstruct_packet = temp_stdout.getvalue()
			reconstruct_packet_s = reconstruct_packet.split(' ')
			proto_name = proto.__name__
			new_chksum = ''
			for i in range(reconstruct_packet_s.index(proto_name)+1, len(reconstruct_packet_s)):
				if 'chksum' in reconstruct_packet_s[i]:
					while new_chksum == '' and i < len(reconstruct_packet_s):
						i += 1
						try:
							new_chksum = int(reconstruct_packet_s[i].strip(),base=16)
						except:
							pass
					break
				if new_chksum != '':
					break
			return old_chksum != new_chksum
		else:
			raise CustomTrafficObjectPickleException('Protocol not contained in the packet')

	def get_stats(self, values, axis=None, out_dim=None):
		if len(values) > 0:
			mean_v = np.mean(values, axis=axis)
			std_v = np.std(values, axis=axis)
			min_v = np.min(values, axis=axis)
			max_v = np.max(values, axis=axis)
			return [mean_v,std_v,min_v,max_v]
		if out_dim is None:
			return [0]*4
		else:
			return [np.zeros((out_dim,))]*4

	def get_by_direction(self, values, dirs, _dir):
		return [ v for i,v in enumerate(values) if dirs[i] == _dir ]

	def UNSWIOT_get_labels(self, smac, dmac):
		label = []
		# Trattare il Gateway come fosse un indirizzo Multicast o Broadcast???
		# Se l'indirizzo MAC sorgente corrisponde a quello di un dispositivo
		if smac in self.labels_dict:
			label = self.labels_dict[smac]
		# Altrimenti, controlla l'indirizzo destinazione e usalo se associato ad un dispositivo
		elif dmac in self.labels_dict:
			label = self.labels_dict[dmac]

		return label

	def BOTIOT_get_labels(self, src_ip, src_port, dst_ip, dst_port, proto, timestamp):
		label = []
		quintuple = '%s,%s,%s,%s,%s' % (src_ip, src_port, dst_ip, dst_port, proto)
		inv_quintuple = '%s,%s,%s,%s,%s' % (dst_ip, dst_port, src_ip, src_port, proto)
		current_quintuple = quintuple
		if current_quintuple not in self.labels_dict:
			if inv_quintuple in self.labels_dict:
				current_quintuple = inv_quintuple
			else:
				return label
		for v in self.labels_dict[current_quintuple]:
			if timestamp >= v[0][0] and timestamp <= v[0][1]:
				label = v[1]
				break
		return label

	def CIC18_get_labels(self, src_ip, dst_ip, timestamp):
		label = [0]+['Normal']*3
		couple = '1%s,%s' % (src_ip, dst_ip)
		inv_couple = '%s,%s' % (dst_ip, src_ip)
		current_couple = couple
		if current_couple not in self.labels_dict:
			if inv_couple in self.labels_dict:
				current_couple = inv_couple
			else:
				return label
		for v in self.labels_dict[current_couple]:
			if timestamp >= v[0][0] and timestamp <= v[0][1]:
				label = v[1]
				break
		return label

	def UNSWIOT_get_unique_label(self, features):
		return features[0][-3:]

	def BOTIOT_get_unique_label(self, features):
		return features[0][-3:]

	def CIC18_get_unique_label(self, features):
		unique_labels,labels_counts = np.unique([ f[-4:] for f in features ], axis=0, return_counts=True)
		# If all the packet of biflow have the same label, return it
		if len(unique_labels) == 1:
			return unique_labels[0]
		# If the packets of biflow does not have the same label, the biflow is anomalous
		if np.array([ul[0]==0 for ul in unique_labels]).any():
			# Removing normal label
			labels_counts = [ lc for lc,ul in zip(labels_counts,unique_labels) if ul[0] != 0 ]
			unique_labels = [ ul for ul in unique_labels if ul[0] != 0 ]
			# If anomalous label is unique, return it
			if len(unique_labels) == 1:
				return unique_labels[0]
		# If there are more than one anomalous label, we elige the most numerous anomaly
		sorting_index = list(reversed(np.argsort(labels_counts)))
		unique_counts = np.unique(labels_counts)
		# Subsequent mechanism permit to select most numerous or randomically among tie
		tie_number = len(unique_counts)
		random_gap = np.random.randint(tie_number)
		return unique_labels[sorting_index[random_gap]]

	# TODO: correct errors
	def CONNECTION_features_extraction_and_labelling(self, only_packets=True):
		for septuple in self.real_traffic_objects:
			for segment in self.real_traffic_objects[septuple]:
				first_packet = segment[0]
				last_packet = segment[-1]
				segment_ftime = first_packet.time
				segment_ltime = last_packet.time
				segment_src = first_packet.src
				segment_dst = first_packet.dst
				label = []
				# Trattare il Gateway come fosse un indirizzo Multicast o Broadcast???
				# Se l'indirizzo MAC sorgente corrisponde a quello di un dispositivo
				if segment_src in self.labels_dict:
					label.append(self.labels_dict[segment_src])
				# Altrimenti, controlla l'indirizzo destinazione e usalo se associato ad un dispositivo
				if segment_dst in self.labels_dict:
					label.append(self.labels_dict[segment_dst])
				# Se va tutto male, salta il segmento
				else:
					continue
				ext_septuple = '-'.join([str(segment_ftime), str(segment_ltime), septuple, '§'.join([ '%'.join(l) for l in label ])])
				cons_pkt = []
				cons_src_pkt = []
				cons_dst_pkt = []
				save_synack = False
				save_ackdat = False
				if ext_septuple not in self.features_per_CONNECTION_packet:
					self.features_per_CONNECTION_packet[ext_septuple] = []
				for pkt in segment:
					dl_src, dl_dst, n_src, n_dst, t_src, t_dst, proto = self.get_septuple(pkt)
					sport = t_src
					dport = t_dst
					if dl_src == segment_src:
						direction = 0
					else:
						direction = 1
					timestamp = pkt.time
					# Compute inter-arrival times
					# Create a list containing the timestamps related to two successive packets
					cons_pkt.append(timestamp)
					if len(cons_pkt) < 2:
						interarrival_time = 0
					else:
						interarrival_time = np.diff(cons_pkt)[0]
						cons_pkt = cons_pkt[1:]
					if direction == 0:
						cons_src_pkt.append(timestamp)
						if len(cons_src_pkt) < 2:
							src_interarrival_time = 0
						else:
							src_interarrival_time = np.diff(cons_src_pkt)[0]
							cons_src_pkt = cons_src_pkt[1:]
					else:
						cons_dst_pkt.append(timestamp)
						if len(cons_dst_pkt) < 2:
							dst_interarrival_time = 0
						else:
							dst_interarrival_time = np.diff(cons_dst_pkt)[0]
							cons_dst_pkt = cons_dst_pkt[1:]
					# Extract the TCP window size
					window_size = 0
					if pkt.haslayer(TCP):
						old_chksum = pkt[TCP].chksum
						del pkt[TCP].chksum
						pkt = IP(str(pkt))
						comp_chksum = pkt[TCP].chksum
						flags = pkt[TCP].flags
						flags = sorted(flags & ( FIN | SYN | RST | PSH | ACK | URG | ECE | CWR ))
						if flags == 'S':
							synack = timestamp
						if flags == 'AS':
							synack = np.abs(synack - timestamp)
							ackdat = timestamp
							save_synack = True
						if ackdat > 1e9 and flags == 'A':
							ackdat = np.abs(ackdat - timestamp)
							save_ackdat = True
						if pkt.haslayer(Raw):
							load = pkt[Raw].load
							load_len = len(load)
						else:
							load = ''
							load_len = 0
						window_size = pkt[TCP].window
					pkt_synack = 0
					pkt_ackdat = 0
					if save_synack:
						pkt_synack = synack
						save_synack = False
					if save_ackdat:
						pkt_ackdat = ackdat
						save_ackdat = False
					if pkt.haslayer(UDP):
						old_chksum = pkt[UDP].chksum
						del pkt[UDP].chksum
						pkt = IP(str(pkt))
						comp_chksum = pkt[UDP].chksum
						if pkt.haslayer(Raw):
							load = pkt[Raw].load
							load_len = len(load)
						else:
							load = ''
							load_len = 0
					load_byte_distr = np.zeros((256,))
					if load_len > 0:
						for b in load:
							load_byte_distr[b] += 1
						load_byte_distr /= load_len
					if old_chksum != comp_chksum and (TCP in pkt or UDP in pkt):
						wrong_fragment = 1
					else:
						wrong_fragment = 0
					features_per_packet = [
						proto,					#0
						sport,					#1
						dport,					#2
						direction,				#3
						interarrival_time,		#4
						src_interarrival_time,	#5
						dts_interarrival_time,	#6
						windows_size,			#7
						flags,					#8
						pkt_synack,				#9
						pkt_ackdat,				#10
						wrong_fragment,			#11
						load_byte_distr,		#12	
						load_len,				#13
						load 					#14
					]
					self.features_per_CONNECTION_packet[ext_septuple].append(features_per_packet)
				if not only_packets:
					curr_features_per_packet = np.asarray(self.features_per_CONNECTION_packet[ext_septuple]).T
					proto = curr_features_per_packet[0][0]
					sport = curr_features_per_packet[1][0]
					dport = curr_features_per_packet[2][0]
					dirs,pkts = np.unique(curr_features_per_packet[3], return_counts=True)
					spkts = pkts[dirs.index(0)]
					dpkts = pkts[dirs.index(1)]
					pkts = spkts+dpkts
					spkts_rate = spkts/pkts
					dpkts_rate = dpkts/pkts
					mean_iat = np.mean(curr_features_per_packet[4])
					std_iat = np.std(curr_features_per_packet[4])
					mean_siat = np.mean(curr_features_per_packet[5])
					std_siat = np.std(curr_features_per_packet[5])
					mean_diat = np.mean(curr_features_per_packet[6])
					std_diat = np.std(curr_features_per_packet[6])
					mean_winsize = np.mean(curr_features_per_packet[7])
					std_winsize = np.std(curr_features_per_packet[7])
					mean_swinsize = np.mean([ ws for i,ws in enumerate(curr_features_per_packet[7]) if curr_features_per_packet[3][i] == 0 ])
					std_swinsize = np.std([ ws for i,ws in enumerate(curr_features_per_packet[7]) if curr_features_per_packet[3][i] == 0 ])
					mean_dwinsize = np.mean([ ws for i,ws in enumerate(curr_features_per_packet[7]) if curr_features_per_packet[3][i] == 1 ])
					std_dwinsize = np.std([ ws for i,ws in enumerate(curr_features_per_packet[7]) if curr_features_per_packet[3][i] == 1 ])
					u_flags, f_count = np.unique(curr_features_per_packet[8],return_counts=True)
					flags = np.zeros((len(STATUS_COMBINATIONS)))
					for f,c in zip(u_flags,f_count):
						flags[STATUS_COMBINATIONS.index(f)] = c
					synack = np.max(curr_features_per_packet[9])
					ackdat = np.max(curr_features_per_packet[10])
					wfrag_sum = np.sum(curr_features_per_packet[11])
					swfrag_sum = np.sum([ wf for i,wf in enumerate(curr_features_per_packet[11]) if curr_features_per_packet[3][i] == 0 ])
					dwfrag_sum = np.sum([ wf for i,wf in enumerate(curr_features_per_packet[11]) if curr_features_per_packet[3][i] == 1 ])
					swfrag_rate = swfrag_sum/wfrag_sum
					dwfrag_rate = dwfrag_sum/wfrag_sum
					mean_load_byte_distr = np.mean(curr_features_per_packet[12], axis=0)
					mean_sload_byte_distr = np.mean([ lbd for i,lbd in enumerate(curr_features_per_packet[12]) if curr_features_per_packet[3][i] == 0 ], axis=0)
					mean_dload_byte_distr = np.mean([ lbd for i,lbd in enumerate(curr_features_per_packet[12]) if curr_features_per_packet[3][i] == 1 ], axis=0)
					std_load_byte_distr = np.std(curr_features_per_packet[12], axis=0)
					std_sload_byte_distr = np.std([ lbd for i,lbd in enumerate(curr_features_per_packet[12]) if curr_features_per_packet[3][i] == 0 ], axis=0)
					std_dload_byte_distr = np.std([ lbd for i,lbd in enumerate(curr_features_per_packet[12]) if curr_features_per_packet[3][i] == 1 ], axis=0)
					mean_load_len = np.mean(curr_features_per_packet[13], axis=0)
					mean_sload_len = np.mean([ ll for i,ll in enumerate(curr_features_per_packet[13]) if curr_features_per_packet[3][i] == 0 ])
					mean_dload_len = np.mean([ ll for i,ll in enumerate(curr_features_per_packet[13]) if curr_features_per_packet[3][i] == 1 ])
					std_load_len = np.std(curr_features_per_packet[13], axis=0)
					std_sload_len = np.std([ ll for i,ll in enumerate(curr_features_per_packet[13]) if curr_features_per_packet[3][i] == 0 ])
					std_dload_len = np.std([ ll for i,ll in enumerate(curr_features_per_packet[13]) if curr_features_per_packet[3][i] == 1 ])
					load_len_sum = np.sum(curr_features_per_packet[13], axis=0)
					load_slen_sum = np.sum([ ll for i,ll in enumerate(curr_features_per_packet[13]) if curr_features_per_packet[3][i] == 0 ])
					load_dlen_sum = np.sum([ ll for i,ll in enumerate(curr_features_per_packet[13]) if curr_features_per_packet[3][i] == 1 ])
					duration = segment_ltime - segment_ftime
					byte_rate = load_len_sum/duration
					sbyte_rate = sload_len_sum/duration
					dbyte_rate = dload_len_sum/duration
					features_per_CONNECTION = [
						proto,
						sport,
						dport,
						pkts,
						spkts_rate,
						dpkts_rate,
						mean_iat,
						std_iat,
						mean_siat,
						std_siat,
						mean_diat,
						std_diat,
						mean_winsize,
						std_winsize,
						mean_swinsize,
						std_swinsize,
						mean_dwinsize,
						std_dwinsize,
						*flags,
						synack,
						ackdat,
						wfrag_sum,
						swfrag_rate,
						dwfrag_rate,
						mean_load_byte_distr,
						std_load_byte_distr,
						mean_sload_byte_distr,
						std_sload_byte_distr,
						mean_dload_byte_distr,
						std_dload_byte_distr,
						mean_load_len,
						std_load_len,
						mean_sload_len,
						std_sload_len,
						mean_dload_len,
						std_dload_len,
						duration,
						byte_rate,
						sbyte_rate,
						dbyte_rate
					]
					self.features_per_CONNECTION[ext_septuple] = features_per_CONNECTION

	# TODO: correct errors
	def BICHANNEL_features_extraction_and_labelling(self, only_packets=True):
		for quadruple in self.real_traffic_objects:
			first_packet = BICHANNEL[0]
			last_packet = BICHANNEL[-1]
			BICHANNEL = self.real_traffic_objects[quadruple]
			BICHANNEL_ftime = first_packet.time
			BICHANNEL_ltime = last_packet.time
			BICHANNEL_src = first_packet.src
			BICHANNEL_dst = first_packet.dst
			label = []
			# Trattare il Gateway come fosse un indirizzo Multicast o Broadcast???
			# Se l'indirizzo MAC sorgente corrisponde a quello di un dispositivo
			if BICHANNEL_src in self.labels_dict:
				label.append(self.labels_dict[BICHANNEL_src])
			# Altrimenti, controlla l'indirizzo destinazione e usalo se associato ad un dispositivo
			if BICHANNEL_dst in self.labels_dict:
				label.append(self.labels_dict[BICHANNEL_dst])
			# Se va tutto male, salta il BICHANNELo
			if len(label) == 0:
				continue
			ext_quadruple = '-'.join([str(BICHANNEL_ftime), str(BICHANNEL_ltime), quadruple, '§'.join([ '%'.join(l) for l in label ])])
			cons_pkt = []
			cons_src_pkt = []
			cons_dst_pkt = []
			if ext_quadruple not in self.features_per_BICHANNEL:
				self.features_per_BICHANNEL[ext_quadruple] = []
			for pkt in BICHANNEL:
				dl_src, dl_dst, n_src, n_dst, t_src, t_dst, proto = self.get_septuple(pkt)
				sport = t_src
				dport = t_dst
				if dl_src == BICHANNEL_src:
					direction = 0
				else:
					direction = 1
				timestamp = pkt.time
				# Compute inter-arrival times
				# Create a list containing the timestamps related to two successive packets
				cons_pkt.append(timestamp)
				if len(cons_pkt) < 2:
					interarrival_time = 0
				else:
					interarrival_time = np.diff(cons_pkt)[0]
					cons_pkt = cons_pkt[1:]
				if direction == 0:
					cons_src_pkt.append(timestamp)
					if len(cons_src_pkt) < 2:
						src_interarrival_time = 0
					else:
						src_interarrival_time = np.diff(cons_src_pkt)[0]
						cons_src_pkt = cons_src_pkt[1:]
				else:
					cons_dst_pkt.append(timestamp)
					if len(cons_dst_pkt) < 2:
						dst_interarrival_time = 0
					else:
						dst_interarrival_time = np.diff(cons_dst_pkt)[0]
						cons_dst_pkt = cons_dst_pkt[1:]
				# Extract the TCP window size
				window_size = 0
				if pkt.haslayer(TCP):
					old_chksum = pkt[TCP].chksum
					del pkt[TCP].chksum
					pkt = IP(str(pkt))
					comp_chksum = pkt[TCP].chksum
					flags = pkt[TCP].flags
					flags = flags & ( FIN | SYN | RST | PSH | ACK | URG | ECE | CWR )
					if pkt.haslayer(Raw):
						load = pkt[Raw].load
						load_len = len(load)
					else:
						load = ''
						load_len = 0
					window_size = pkt[TCP].window
				if pkt.haslayer(UDP):
					old_chksum = pkt[UDP].chksum
					del pkt[UDP].chksum
					pkt = IP(str(pkt))
					comp_chksum = pkt[UDP].chksum
					if pkt.haslayer(Raw):
						load = pkt[Raw].load
						load_len = len(load)
					else:
						load = ''
						load_len = 0
				load_byte_distr = np.zeros((256,))
				if load_len > 0:
					for b in load:
						load_byte_distr[b] += 1
					load_byte_distr /= load_len
				if old_chksum != comp_chksum and (TCP in pkt or UDP in pkt):
					wrong_fragment = 1
				else:
					wrong_fragment = 0
				features_per_packet = [
					proto,					#0
					sport,					#1
					dport,					#2
					direction,				#3
					interarrival_time,		#4
					src_interarrival_time,	#5
					dts_interarrival_time,	#6
					windows_size,			#7
					flags,					#8
					wrong_fragment,			#9
					load_byte_distr,		#10	
					load_len,				#11
					load, 					#12
					timestamp 				#13
				]
				self.features_per_BICHANNEL_packet[ext_quadruple].append(features_per_packet)
			if not only_packets:
				curr_features_per_packet = np.asarray(self.features_per_BICHANNEL_packet[ext_quadruple]).T
				u_protos, p_count = np.unique(curr_features_per_packet[0],return_counts=True)
				protos = np.zeros((len(PROTOS)))
				for p,c in zip(u_protos,p_count):
					protos[PROTOS.index(p)] = c
				u_sports, s_count = np.unique(curr_features_per_packet[1],return_counts=True)
				sports = sorted(zip(u_sports, s_count))
				u_dports, d_count = np.unique(curr_features_per_packet[2],return_counts=True)
				dports = sorted(zip(u_dports, d_count))
				dirs,pkts = np.unique(curr_features_per_packet[3], return_counts=True)
				spkts = pkts[dirs.index(0)]
				dpkts = pkts[dirs.index(1)]
				pkts = spkts+dpkts
				spkts_rate = spkts/pkts
				dpkts_rate = dpkts/pkts
				mean_iat = np.mean(curr_features_per_packet[4])
				std_iat = np.std(curr_features_per_packet[4])
				mean_siat = np.mean(curr_features_per_packet[5])
				std_siat = np.std(curr_features_per_packet[5])
				mean_diat = np.mean(curr_features_per_packet[6])
				std_diat = np.std(curr_features_per_packet[6])
				mean_winsize = np.mean(curr_features_per_packet[7])
				std_winsize = np.std(curr_features_per_packet[7])
				mean_swinsize = np.mean([ ws for i,ws in enumerate(curr_features_per_packet[7]) if curr_features_per_packet[3][i] == 0 ])
				std_swinsize = np.std([ ws for i,ws in enumerate(curr_features_per_packet[7]) if curr_features_per_packet[3][i] == 0 ])
				mean_dwinsize = np.mean([ ws for i,ws in enumerate(curr_features_per_packet[7]) if curr_features_per_packet[3][i] == 1 ])
				std_dwinsize = np.std([ ws for i,ws in enumerate(curr_features_per_packet[7]) if curr_features_per_packet[3][i] == 1 ])
				u_flags, f_count = np.unique(curr_features_per_packet[8],return_counts=True)
				flags = np.zeros((len(STATUS_COMBINATIONS)))
				for f,c in zip(u_flags,f_count):
					flags[STATUS_COMBINATIONS.index(f)] = c
				wfrag_sum = np.sum(curr_features_per_packet[9])
				swfrag_sum = np.sum([ wf for i,wf in enumerate(curr_features_per_packet[9]) if curr_features_per_packet[3][i] == 0 ])
				dwfrag_sum = np.sum([ wf for i,wf in enumerate(curr_features_per_packet[9]) if curr_features_per_packet[3][i] == 1 ])
				swfrag_rate = swfrag_sum/wfrag_sum
				dwfrag_rate = dwfrag_sum/wfrag_sum
				mean_load_byte_distr = np.mean(curr_features_per_packet[10], axis=0)
				mean_sload_byte_distr = np.mean([ lbd for i,lbd in enumerate(curr_features_per_packet[10]) if curr_features_per_packet[3][i] == 0 ], axis=0)
				mean_dload_byte_distr = np.mean([ lbd for i,lbd in enumerate(curr_features_per_packet[10]) if curr_features_per_packet[3][i] == 1 ], axis=0)
				std_load_byte_distr = np.std(curr_features_per_packet[10], axis=0)
				std_sload_byte_distr = np.std([ lbd for i,lbd in enumerate(curr_features_per_packet[10]) if curr_features_per_packet[3][i] == 0 ], axis=0)
				std_dload_byte_distr = np.std([ lbd for i,lbd in enumerate(curr_features_per_packet[10]) if curr_features_per_packet[3][i] == 1 ], axis=0)
				mean_load_len = np.mean(curr_features_per_packet[11], axis=0)
				mean_sload_len = np.mean([ ll for i,ll in enumerate(curr_features_per_packet[11]) if curr_features_per_packet[3][i] == 0 ])
				mean_dload_len = np.mean([ ll for i,ll in enumerate(curr_features_per_packet[11]) if curr_features_per_packet[3][i] == 1 ])
				std_load_len = np.std(curr_features_per_packet[11], axis=0)
				std_sload_len = np.std([ ll for i,ll in enumerate(curr_features_per_packet[11]) if curr_features_per_packet[3][i] == 0 ])
				std_dload_len = np.std([ ll for i,ll in enumerate(curr_features_per_packet[11]) if curr_features_per_packet[3][i] == 1 ])
				load_len_sum = np.sum(curr_features_per_packet[11], axis=0)
				load_slen_sum = np.sum([ ll for i,ll in enumerate(curr_features_per_packet[11]) if curr_features_per_packet[3][i] == 0 ])
				load_dlen_sum = np.sum([ ll for i,ll in enumerate(curr_features_per_packet[11]) if curr_features_per_packet[3][i] == 1 ])
				duration = BICHANNEL_ltime - BICHANNEL_ftime
				byte_rate = load_len_sum/duration
				sbyte_rate = sload_len_sum/duration
				dbyte_rate = dload_len_sum/duration
				features_per_BICHANNEL = [
					protos,
					sports,
					dports,
					pkts,
					spkts_rate,
					dpkts_rate,
					mean_iat,
					std_iat,
					mean_siat,
					std_siat,
					mean_diat,
					std_diat,
					mean_winsize,
					std_winsize,
					mean_swinsize,
					std_swinsize,
					mean_dwinsize,
					std_dwinsize,
					flags,
					wfrag_sum,
					swfrag_rate,
					dwfrag_rate,
					mean_load_byte_distr,
					std_load_byte_distr,
					mean_sload_byte_distr,
					std_sload_byte_distr,
					mean_dload_byte_distr,
					std_dload_byte_distr,
					mean_load_len,
					std_load_len,
					mean_sload_len,
					std_sload_len,
					mean_dload_len,
					std_dload_len,
					duration,
					byte_rate,
					sbyte_rate,
					dbyte_rate
				]
				self.features_per_BICHANNEL[ext_quadruple] = features_per_BICHANNEL

	def BIFLOW_features_extraction_and_labelling(self, only_packets=True):
		import time
		import progressbar

		self.per_packet_attributes = [
			('proto',u'STRING'),('sport',u'STRING'),('dport',u'STRING'),('direction',u'REAL'),
			('interarrival_time',u'REAL'),('src_interarrival_time',u'REAL'),('dst_interarrival_time',u'REAL'),
			('window_size',u'REAL'),('flags',u'STRING'),('wrong_fragment',u'REAL'),('load_byte_distr',u'SPARRAY'),
			('load_len',u'REAL'),('load',u'STRING'),('ttl',u'REAL'),('icmp_type',u'STRING'),('timestamp',u'REAL')
		]
		self.per_traffic_object_attributes = [
			('proto',u'STRING'),
			# ('sport',u'STRING'),
			('dport',u'STRING'),('pkts',u'REAL'),('spkts_rate',u'REAL'),('dpkts_rate',u'REAL'),
			('mean_iat',u'REAL'),('std_iat',u'REAL'),('min_iat',u'REAL'),('max_iat',u'REAL'),
			('mean_siat',u'REAL'),('std_siat',u'REAL'),('min_siat',u'REAL'),('max_siat',u'REAL'),
			('mean_diat',u'REAL'),('std_diat',u'REAL'),('min_diat',u'REAL'),('max_diat',u'REAL'),
			('mean_winsize',u'REAL'),('std_winsize',u'REAL'),('min_winsize',u'REAL'),('max_winsize',u'REAL'),
			('mean_swinsize',u'REAL'),('std_swinsize',u'REAL'),('min_swinsize',u'REAL'),('max_swinsize',u'REAL'),
			('mean_dwinsize',u'REAL'),('std_dwinsize',u'REAL'),('min_dwinsize',u'REAL'),('max_dwinsize',u'REAL'),
			('wfrag_sum',u'REAL'),('swfrag_rate',u'REAL'),('dwfrag_rate',u'REAL'),
			('mean_load_byte_distr',u'SPARRAY'),
			# ('std_load_byte_distr',u'SPARRAY'),('min_load_byte_distr',u'SPARRAY'),('max_load_byte_distr',u'SPARRAY'),
			('mean_sload_byte_distr',u'SPARRAY'),
			# ('std_sload_byte_distr',u'SPARRAY'),('min_sload_byte_distr',u'SPARRAY'),('max_sload_byte_distr',u'SPARRAY'),
			('mean_dload_byte_distr',u'SPARRAY'),
			# ('std_dload_byte_distr',u'SPARRAY'),('min_dload_byte_distr',u'SPARRAY'),('max_dload_byte_distr',u'SPARRAY'),
			('mean_load_len',u'REAL'),('std_load_len',u'REAL'),('min_load_len',u'REAL'),('max_load_len',u'REAL'),
			('mean_sload_len',u'REAL'),('std_sload_len',u'REAL'),('min_sload_len',u'REAL'),('max_sload_len',u'REAL'),
			('mean_dload_len',u'REAL'),('std_dload_len',u'REAL'),('min_dload_len',u'REAL'),('max_dload_len',u'REAL'),
			('duration',u'REAL'),('byte_rate',u'REAL'),('sbyte_rate',u'REAL'),('dbyte_rate',u'REAL'),
			('mean_ttl',u'REAL'),('std_ttl',u'REAL'),('min_ttl',u'REAL'),('max_ttl',u'REAL'),
			('mean_sttl',u'REAL'),('std_sttl',u'REAL'),('min_sttl',u'REAL'),('max_sttl',u'REAL'),
			('mean_dttl',u'REAL'),('std_dttl',u'REAL'),('min_dttl',u'REAL'),('max_dttl',u'REAL'),
			('flags',u'SPARRAY'),('icmp_types',u'SPARRAY')
		]
		self.label_attributes = [('label_0',u'STRING'),('label_1',u'STRING'),('label_2',u'STRING')]
		if self.dataset_name == 'CIC18':
			self.label_attributes.append(('label_3',u'STRING'))

		# for _,septuple in zip(progressbar.progressbar(range(len(self.real_traffic_objects)), redirect_stdout=True),self.real_traffic_objects):
		for septuple in self.real_traffic_objects:
			traffic_object = self.real_traffic_objects[septuple]
			# print(septuple,len(traffic_object))
			first_packet = traffic_object[0]
			last_packet = traffic_object[-1]
			traffic_object_ftime = first_packet.time
			traffic_object_ltime = last_packet.time
			traffic_object_src = first_packet.src
			traffic_object_dst = first_packet.dst

			if self.dataset_name == 'UNSWIOT':
				label = self.factory_methods('d', 'get_labels', [traffic_object_src,traffic_object_dst ])
				if len(label) == 0:
					continue
			# # Se va tutto male, salta il traffic_object
			ext_septuple = '-'.join([str(traffic_object_ftime), str(traffic_object_ltime), septuple])
			cons_pkt = []
			cons_src_pkt = []
			cons_dst_pkt = []
			if ext_septuple not in self.dataset_per_packet:
				self.dataset_per_packet[ext_septuple] = []
			for pkt in traffic_object:
				dl_src, dl_dst, n_src, n_dst, t_src, t_dst, proto = self.get_septuple(pkt)
				sport = t_src
				dport = t_dst
				if dl_src == traffic_object_src:
					direction = 0
				else:
					direction = 1
				timestamp = pkt.time
				if self.dataset_name == 'BOTIOT':
					label = self.factory_methods('d', 'get_labels', [ *np.asarray(septuple.split(','))[[1,2,4,5,6]], timestamp ])
					if len(label) == 0:
						continue
				if self.dataset_name == 'CIC18':
					label = self.factory_methods('d', 'get_labels', [ n_src, n_dst, timestamp ])
					if len(label) == 0:
						continue
				# Compute inter-arrival times
				# Create a list containing the timestamps related to two successive packets
				cons_pkt.append(timestamp)
				interarrival_time = 0
				src_interarrival_time = 0
				dst_interarrival_time = 0
				if len(cons_pkt) >= 2:
					interarrival_time = np.diff(cons_pkt)[0]
					cons_pkt = cons_pkt[1:]
				if direction == 0:
					cons_src_pkt.append(timestamp)
					if len(cons_src_pkt) >= 2:
						src_interarrival_time = np.diff(cons_src_pkt)[0]
						cons_src_pkt = cons_src_pkt[1:]
				else:
					cons_dst_pkt.append(timestamp)
					if len(cons_dst_pkt) >= 2:
						dst_interarrival_time = np.diff(cons_dst_pkt)[0]
						cons_dst_pkt = cons_dst_pkt[1:]
				# Extract the TCP window size
				window_size = 0
				wrong_fragment = 0
				icmp_type = ''
				load = ''
				flags = ''
				load_len = 0
				if pkt.haslayer(TCP):
					wrong_fragment = [ 1 if self.check_chksum(TCP, pkt) else 0 ][0]
					flags = pkt[TCP].flags
					flags = flags & ( FIN | SYN | RST | PSH | ACK | URG | ECE | CWR )
					if pkt.haslayer(Raw):
						load = pkt[Raw].load
						load_len = len(load)
					else:
						load = ''
						load_len = 0
					window_size = pkt[TCP].window
				elif pkt.haslayer(UDP) and not pkt.haslayer(ICMP):
					wrong_fragment = [ 1 if self.check_chksum(UDP, pkt) else 0 ][0]
					if pkt.haslayer(Raw):
						load = pkt[Raw].load
						load_len = len(load)
					else:
						load = ''
						load_len = 0
				elif pkt.haslayer(ICMP):
					icmp_type = pkt[ICMP].type
				load_byte_distr = np.zeros((256,))
				if type(load_len) == int and load_len > 0:
					for b in load:
						load_byte_distr[b] += 1
					load_byte_distr /= load_len
				ttl = 0
				if IP in pkt:
					ttl=pkt.ttl
				elif IPv6 in pkt:
					ttl=pkt.hlim
				features_per_packet = [
					proto,					#0
					sport,					#1
					dport,					#2
					direction,				#3
					interarrival_time,		#4
					src_interarrival_time,	#5
					dst_interarrival_time,	#6
					window_size,			#7
					flags,					#8
					wrong_fragment,			#9
					csr_matrix(load_byte_distr),		#10	
					load_len,				#11
					load, 					#12
					ttl, 					#13
					icmp_type,				#14
					timestamp,
					*label
				]
				self.dataset_per_packet[ext_septuple].append(features_per_packet)
			if not only_packets and len(self.dataset_per_packet[ext_septuple]) > 0:
				curr_features_per_packet = np.asarray(self.dataset_per_packet[ext_septuple]).T
				proto = curr_features_per_packet[0][0]
				sport = curr_features_per_packet[1][0]
				dport = curr_features_per_packet[2][0]
				dirs,pkts_per_dir = np.unique(curr_features_per_packet[3], return_counts=True)
				spkts = 0
				dpkts = 0
				if 0 in dirs:
					spkts = pkts_per_dir[np.where(dirs == 0)][0]
				if 1 in dirs:
					dpkts = pkts_per_dir[np.where(dirs == 1)][0]
				pkts = spkts+dpkts
				spkts_rate = 0
				dpkts_rate = 0
				if pkts != 0:
					spkts_rate = spkts/pkts
					dpkts_rate = dpkts/pkts
				iat = self.get_stats(curr_features_per_packet[4])
				siat = self.get_stats(curr_features_per_packet[5])
				diat = self.get_stats(curr_features_per_packet[6])
				winsize = self.get_stats(curr_features_per_packet[7])
				swinsize = self.get_stats(self.get_by_direction(curr_features_per_packet[7],curr_features_per_packet[3],0))
				dwinsize = self.get_stats(self.get_by_direction(curr_features_per_packet[7],curr_features_per_packet[3],1))

				u_flags, f_counts = np.unique(curr_features_per_packet[8],return_counts=True)
				flags = np.zeros((len(STATUS_COMBINATIONS)))
				for f,c in zip(u_flags,f_counts):
					flags[STATUS_COMBINATIONS.index(f)] = c
				flags /= np.sum(f_counts)

				wfrag_sum = np.sum(curr_features_per_packet[9])
				swfrag_sum = np.sum([ wf for i,wf in enumerate(curr_features_per_packet[9]) if curr_features_per_packet[3][i] == 0 ])
				dwfrag_sum = np.sum([ wf for i,wf in enumerate(curr_features_per_packet[9]) if curr_features_per_packet[3][i] == 1 ])
				swfrag_rate = 0
				dwfrag_rate = 0
				if wfrag_sum != 0:
					swfrag_rate = swfrag_sum/wfrag_sum
					dwfrag_rate = dwfrag_sum/wfrag_sum
				
				load_byte_distr = self.get_stats([ lbd.todense() for lbd in curr_features_per_packet[10] ], axis=0)
				sload_byte_distr = self.get_stats([ lbd.todense() for i,lbd in enumerate(curr_features_per_packet[10]) if curr_features_per_packet[3][i] == 0 ], axis=0, out_dim=256)
				dload_byte_distr = self.get_stats([ lbd.todense() for i,lbd in enumerate(curr_features_per_packet[10]) if curr_features_per_packet[3][i] == 1 ], axis=0, out_dim=256)

				load_len = self.get_stats(curr_features_per_packet[11])
				sload_len = self.get_stats(self.get_by_direction(curr_features_per_packet[11], curr_features_per_packet[3], 0))
				dload_len = self.get_stats(self.get_by_direction(curr_features_per_packet[11], curr_features_per_packet[3], 1))

				load_len_sum = np.sum(curr_features_per_packet[11], axis=0)
				sload_len_sum = np.sum([ ll for i,ll in enumerate(curr_features_per_packet[11]) if curr_features_per_packet[3][i] == 0 ])
				dload_len_sum = np.sum([ ll for i,ll in enumerate(curr_features_per_packet[11]) if curr_features_per_packet[3][i] == 1 ])
				duration = traffic_object_ltime - traffic_object_ftime
				byte_rate = 0
				sbyte_rate = 0
				dbyte_rate = 0
				if duration != 0:
					byte_rate = load_len_sum/duration
					sbyte_rate = sload_len_sum/duration
					dbyte_rate = dload_len_sum/duration

				ttl = self.get_stats(curr_features_per_packet[13])
				sttl = self.get_stats(self.get_by_direction(curr_features_per_packet[13], curr_features_per_packet[3], 0))
				dttl = self.get_stats(self.get_by_direction(curr_features_per_packet[13], curr_features_per_packet[3], 1))

				u_icmptypes, t_counts = np.unique(curr_features_per_packet[14],return_counts=True)
				icmp_types = np.zeros((len(ICMP_CODES)))
				for t,c in zip(u_icmptypes,t_counts):
					if t != '':
						icmp_types[ICMP_CODES.index(t)] = c
					else:
						icmp_types[-1] = c
				icmp_types /= np.sum(t_counts)

				labels_per_traffic_object = self.factory_methods('d', 'get_unique_label', [ self.dataset_per_packet[ext_septuple] ])

				features_per_traffic_object = [
					proto,		# Protocol is useful to distinguish among flows. Some
					# sport,
					dport,
					pkts,
					spkts_rate,
					dpkts_rate,
					*iat,
					*siat,
					*diat,
					*winsize,
					*swinsize,
					*dwinsize,
					wfrag_sum,
					swfrag_rate,
					dwfrag_rate,
					csr_matrix(load_byte_distr[0]),
					csr_matrix(sload_byte_distr[0]),
					csr_matrix(dload_byte_distr[0]),
					*load_len,
					*sload_len,
					*dload_len,
					duration,
					byte_rate,
					sbyte_rate,
					dbyte_rate,
					*ttl,
					*sttl,
					*dttl,
					csr_matrix(flags),
					csr_matrix(icmp_types),
					*labels_per_traffic_object
				]

				self.dataset_per_traffic_object[ext_septuple] = features_per_traffic_object

	def BAG_OF_SENT_PACKETS_features_extraction_and_labelling(self, only_packets=True):

		self.per_packet_attributes = [
			('interarrival_time',u'REAL'),
			('load_len',u'REAL'),
			('timestamp',u'REAL'),
			('labels','SEQ')
		]
		self.per_traffic_object_attributes = [
			('pkts',u'REAL'),
			('mean_iat',u'REAL'),('std_iat',u'REAL'),('min_iat',u'REAL'),('max_iat',u'REAL'),
			('mean_load_len',u'REAL'),('std_load_len',u'REAL'),('min_load_len',u'REAL'),('max_load_len',u'REAL'),
			('duration',u'REAL'),
			('byte_rate',u'REAL')
		]

		# for _,src_mac in zip(progressbar.progressbar(range(len(self.real_traffic_objects)), redirect_stdout=True),self.real_traffic_objects):
		for src_mac in self.real_traffic_objects:
			if src_mac not in self.dataset_per_packet:
				self.dataset_per_packet[src_mac] = []
			for traffic_object in self.real_traffic_objects[src_mac]:
				self.dataset_per_packet[src_mac].append([])
				# print(src_mac,len(traffic_object))
				first_packet = traffic_object[0]
				last_packet = traffic_object[-1]
				# traffic_object_ftime = first_packet.time
				# traffic_object_ltime = last_packet.time
				traffic_object_src = first_packet.src
				traffic_object_dst = first_packet.dst
				if self.dataset_name == 'UNSWIOT':
					label = self.factory_methods('d', 'get_labels', [traffic_object_src,None])
					if len(label) == 0:
						continue
				# # Se va tutto male, salta il traffic_object
				for pkt in traffic_object:
					dl_src, dl_dst, n_src, n_dst, t_src, t_dst, proto = self.get_septuple(pkt)
					timestamp = pkt.time
					load_len = 0
					if (pkt.haslayer(TCP) or pkt.haslayer(UDP)) and pkt.haslayer(Raw):
						load_len = len(pkt[Raw].load)
					if self.dataset_name == 'BOTIOT':
						label = self.factory_methods('d', 'get_labels', [ *np.asarray(src_mac.split(','))[[1,2,4,5]], timestamp ])
						if len(label) == 0:
							continue
					features_per_packet = np.asarray([
						timestamp,				#0
						load_len,				#1
						*label 					#2
					], dtype=object)
					self.dataset_per_packet[src_mac][-1].append(features_per_packet)
		for src_mac in self.dataset_per_packet:
			if src_mac not in self.dataset_per_traffic_object:
				self.dataset_per_traffic_object[src_mac] = []
			for mult in [1,5,15,100,600]:
				self.dataset_per_traffic_object[src_mac].append([])
				iter_dataset = iter(self.dataset_per_packet[src_mac])
				for traffic_object in iter_dataset:
					for i in range(mult-1):
						try:
							traffic_object += next(iter_dataset)
						except:
							break
					if not only_packets and len(traffic_object) > 0:
						curr_features_per_packet = np.asarray(traffic_object).T
						pkts = len(traffic_object)
						timestamps = sorted([curr_features_per_packet[0][0]] + list(curr_features_per_packet[0]))
						iats = [ timestamps[i] - timestamps[i-1] for i in range(1,len(timestamps)) ]
						# if mult == 5:
						# 	print(iats)
						# 	input()
						iat = self.get_stats(iats)
						load_lens = curr_features_per_packet[1]
						load_len = self.get_stats(load_lens)
						load_len_sum = np.sum(load_lens, axis=0)
						duration = timestamps[-1] - timestamps[0]
						byte_rate = 0
						if duration != 0:
							byte_rate = load_len_sum/duration
						labels = curr_features_per_packet[2:,0]
						features_per_traffic_object = [
							pkts,
							*iat,
							*load_len,
							duration,
							byte_rate,
							*labels
						]
						self.dataset_per_traffic_object[src_mac][-1].append(features_per_traffic_object)

	def BAG_OF_RECEIVED_PACKETS_features_extraction_and_labelling(self, only_packets=True):
		import time
		import progressbar

		self.per_packet_attributes = [
			('proto',u'STRING'),('sport',u'STRING'),('dport',u'STRING'),('interarrival_time',u'REAL'),
			('window_size',u'REAL'),('flags',u'STRING'),('wrong_fragment',u'REAL'),('load_byte_distr',u'SPARRAY'),
			('load_len',u'REAL'),('load',u'STRING'),('ttl',u'REAL'),('icmp_type',u'STRING'),('timestamp',u'REAL')
		]
		self.per_traffic_object_attributes = [
			('proto',u'SPARRAY'),
			# ('sport',u'STRING'),('dport',u'STRING'),
			('pkts',u'REAL'),
			# ('spkts_rate',u'REAL'),('dpkts_rate',u'REAL'),
			('mean_iat',u'REAL'),('std_iat',u'REAL'),('min_iat',u'REAL'),('max_iat',u'REAL'),
			# ('mean_siat',u'REAL'),('std_siat',u'REAL'),('min_siat',u'REAL'),('max_siat',u'REAL'),
			# ('mean_diat',u'REAL'),('std_diat',u'REAL'),('min_diat',u'REAL'),('max_diat',u'REAL'),
			('mean_winsize',u'REAL'),('std_winsize',u'REAL'),('min_winsize',u'REAL'),('max_winsize',u'REAL'),
			# ('mean_swinsize',u'REAL'),('std_swinsize',u'REAL'),('min_swinsize',u'REAL'),('max_swinsize',u'REAL'),
			# ('mean_dwinsize',u'REAL'),('std_dwinsize',u'REAL'),('min_dwinsize',u'REAL'),('max_dwinsize',u'REAL'),
			('wfrag_sum',u'REAL'),
			('swfrag_rate',u'REAL'),('dwfrag_rate',u'REAL'),
			('mean_load_byte_distr',u'SPARRAY'),('std_load_byte_distr',u'SPARRAY'),
			('min_load_byte_distr',u'SPARRAY'),('max_load_byte_distr',u'SPARRAY'),
			# ('mean_sload_byte_distr',u'SPARRAY'),('std_sload_byte_distr',u'SPARRAY'),
			# ('min_sload_byte_distr',u'SPARRAY'),('max_sload_byte_distr',u'SPARRAY'),
			# ('mean_dload_byte_distr',u'SPARRAY'),('std_dload_byte_distr',u'SPARRAY'),
			# ('min_dload_byte_distr',u'SPARRAY'),('max_dload_byte_distr',u'SPARRAY'),
			('mean_load_len',u'REAL'),('std_load_len',u'REAL'),('min_load_len',u'REAL'),('max_load_len',u'REAL'),
			# ('mean_sload_len',u'REAL'),('std_sload_len',u'REAL'),('min_sload_len',u'REAL'),('max_sload_len',u'REAL'),
			# ('mean_dload_len',u'REAL'),('std_dload_len',u'REAL'),('min_dload_len',u'REAL'),('max_dload_len',u'REAL'),
			('duration',u'REAL'),
			('byte_rate',u'REAL'),
			# ('sbyte_rate',u'REAL'),('dbyte_rate',u'REAL'),
			('mean_ttl',u'REAL'),('std_ttl',u'REAL'),('min_ttl',u'REAL'),('max_ttl',u'REAL'),
			# ('mean_sttl',u'REAL'),('std_sttl',u'REAL'),('min_sttl',u'REAL'),('max_sttl',u'REAL'),
			# ('mean_dttl',u'REAL'),('std_dttl',u'REAL'),('min_dttl',u'REAL'),('max_dttl',u'REAL'),
			('flags',u'SPARRAY'),('icmp_types',u'SPARRAY')
		]

		# for _,ip_src in zip(progressbar.progressbar(range(len(self.real_traffic_objects)), redirect_stdout=True),self.real_traffic_objects):
		for ip_src in self.real_traffic_objects:
			for traffic_object in self.real_traffic_objects[ip_src]:
				# print(ip_src,len(traffic_object))
				first_packet = traffic_object[0]
				last_packet = traffic_object[-1]
				traffic_object_ftime = first_packet.time
				traffic_object_ltime = last_packet.time
				traffic_object_src = first_packet.src
				traffic_object_dst = first_packet.dst

				if self.dataset_name == 'UNSWIOT':
					label = self.factory_methods('d', 'get_labels', [traffic_object_src,traffic_object_dst ])
					if len(label) == 0:
						continue
				# # Se va tutto male, salta il traffic_object
				ext_ip_src = '-'.join([str(traffic_object_ftime), str(traffic_object_ltime), ip_src])
				cons_pkt = []
				if ext_ip_src not in self.dataset_per_packet:
					self.dataset_per_packet[ext_ip_src] = []
					# Array for features
					self.dataset_per_packet[ext_ip_src].append([])
					# Array for labels
					self.dataset_per_packet[ext_ip_src].append([])
				for pkt in traffic_object[:MAX_PKT_PER_TO]:
					dl_src, dl_dst, n_src, n_dst, t_src, t_dst, proto = self.get_septuple(pkt)
					sport = t_src
					dport = t_dst
					timestamp = pkt.time
					if self.dataset_name == 'BOTIOT':
						label = self.factory_methods('d', 'get_labels', [ *np.asarray(ip_src.split(','))[[1,2,4,5]], timestamp ])
						if len(label) == 0:
							continue
					# Compute inter-arrival times
					# Create a list containing the timestamps related to two successive packets
					cons_pkt.append(timestamp)
					interarrival_time = 0
					if len(cons_pkt) >= 2:
						interarrival_time = np.diff(cons_pkt)[0]
						cons_pkt = cons_pkt[1:]
					# Extract the TCP window size
					window_size = 0
					wrong_fragment = 0
					icmp_type = ''
					load = ''
					flags = ''
					load_len = 0
					if pkt.haslayer(TCP):
						wrong_fragment = [ 1 if self.check_chksum(TCP, pkt) else 0 ][0]
						flags = pkt[TCP].flags
						flags = flags & ( FIN | SYN | RST | PSH | ACK | URG | ECE | CWR )
						if pkt.haslayer(Raw):
							load = pkt[Raw].load
							load_len = len(load)
						else:
							load = ''
							load_len = 0
						window_size = pkt[TCP].window
					elif pkt.haslayer(UDP) and not pkt.haslayer(ICMP):
						wrong_fragment = [ 1 if self.check_chksum(UDP, pkt) else 0 ][0]
						if pkt.haslayer(Raw):
							load = pkt[Raw].load
							load_len = len(load)
						else:
							load = ''
							load_len = 0
					elif pkt.haslayer(ICMP):
						icmp_type = pkt[ICMP].type
					load_byte_distr = np.zeros((256,))
					if type(load_len) == int and load_len > 0:
						for b in load:
							load_byte_distr[b] += 1
						load_byte_distr /= load_len
					ttl = 0
					if IP in pkt:
						ttl=pkt.ttl
					elif IPv6 in pkt:
						ttl=pkt.hlim
					features_per_packet = [
						proto,					#0
						sport,					#1
						dport,					#2
						interarrival_time,		#3
						window_size,			#4
						flags,					#5
						wrong_fragment,			#6
						csr_matrix(load_byte_distr),		#7	
						load_len,				#8
						load, 					#9
						ttl, 					#10
						icmp_type,				#11
						timestamp
					]
					labels_per_packet = [
						*label
					]
					self.dataset_per_packet[ext_ip_src][0].append(features_per_packet)
					self.dataset_per_packet[ext_ip_src][1].append(labels_per_packet)
				if not only_packets and len(self.dataset_per_packet[ext_ip_src][0]) > 0:
					curr_features_per_packet = np.asarray(self.dataset_per_packet[ext_ip_src][0]).T
					u_protos, p_count = np.unique(curr_features_per_packet[0],return_counts=True)
					protos = np.zeros((len(PROTOS)))
					for p,c in zip(u_flags,f_count):
						if p != '':
							protos[PROTOS.index(p)] = c
					sport = curr_features_per_packet[1][0]
					dport = curr_features_per_packet[2][0]
					pkts = len(self.dataset_per_packet[ext_ip_src][0])
					iat = self.get_stats(curr_features_per_packet[3])
					winsize = self.get_stats(curr_features_per_packet[4])

					u_flags, f_count = np.unique(curr_features_per_packet[5],return_counts=True)
					flags = np.zeros((len(STATUS_COMBINATIONS)))
					for f,c in zip(u_flags,f_count):
						if f != '':
							flags[STATUS_COMBINATIONS.index(f)] = c

					wfrag_sum = np.sum(curr_features_per_packet[6])
					
					load_byte_distr = self.get_stats([ lbd.todense() for lbd in curr_features_per_packet[7] ], axis=0)

					load_len = self.get_stats(curr_features_per_packet[8])

					load_len_sum = np.sum(curr_features_per_packet[8], axis=0)
					duration = traffic_object_ltime - traffic_object_ftime
					byte_rate = 0
					if duration != 0:
						byte_rate = load_len_sum/duration

					ttl = self.get_stats(curr_features_per_packet[10])

					u_icmptypes, t_count = np.unique([ icmp_type for icmp_type in curr_features_per_packet[11] if icmp_type != '' ],return_counts=True)
					icmp_types = np.zeros((len(ICMP_CODES)))
					for t,c in zip(u_icmptypes,t_count):
						if t != '':
							icmp_types[ICMP_CODES.index(t)] = c

					features_per_traffic_object = [
						csr_matrix(proto),
						# sport,
						# dport,
						pkts,
						*iat,
						*winsize,
						wfrag_sum,
						*[csr_matrix(stat) for stat in load_byte_distr],
						*load_len,
						duration,
						byte_rate,
						*ttl,
						csr_matrix(flags),
						csr_matrix(icmp_types)
					]
					labels_per_traffic_object = self.dataset_per_packet[ext_ip_src][0][0]

					if ext_ip_src not in self.dataset_per_traffic_object:
						self.dataset_per_traffic_object[ext_ip_src] = []
					self.dataset_per_traffic_object[ext_ip_src].append(features_per_traffic_object)
					self.dataset_per_traffic_object[ext_ip_src].append(labels_per_traffic_object)

	def prepare_dataset_for_dump(self):
		# if not os.path.isdir('./global_CSVs/'):
		# 	os.mkdir('./global_CSVs/')
		# counter = 0
		# counter_str = '000'
		# new_out_csv = './global_CSVs/' + self.out_pickle.split('.')[0] + '_' + counter_str

		# for i,mult in enumerate([1,5,15,100,600]):
		# 	with open('%s_%s_L%s_traffic_objects.csv' % (new_out_csv, self.traffic_object_name, mult), 'w') as csv_dataset_traffic_objects:
		# 		csv_dataset_traffic_objects.write(','.join([ 'L%s_%s' % (mult, a[0]) for a in self.per_traffic_object_attributes ])+',')
		# 		csv_dataset_traffic_objects.write('connection_type,device_class,device')
		# 		for ip_src in sorted(list(self.dataset_per_traffic_object.keys())):
		# 			# for tw in self.dataset_per_traffic_object[ip_src]:
		# 			tw = self.dataset_per_traffic_object[ip_src][i]
		# 				# with open('%s_%s_packets.csv' % (new_out_csv, self.traffic_object_name), 'w') as csv_dataset_packets,\
		# 			csv_dataset_traffic_objects.write('\n'+'\n'.join([','.join([ str(f) for f in ff ]) for ff in tw]))
		# 				# csv_dataset_packets.close()
		# 		csv_dataset_traffic_objects.close()
		self.packets_to_dump = {}
		self.traffic_objects_to_dump = {}

		self.packets_to_dump['attributes'] = [('to_ID',u'STRING')] + self.per_packet_attributes + self.label_attributes
		self.traffic_objects_to_dump['attributes'] = [('to_ID',u'STRING')] + self.per_traffic_object_attributes + self.label_attributes
		self.packets_to_dump['data'] = []
		self.traffic_objects_to_dump['data'] = []

		for ext_septuple in self.dataset_per_packet:
			if len(self.dataset_per_packet[ext_septuple]) > 0:
				for features_per_packet in self.dataset_per_packet[ext_septuple]:
					self.packets_to_dump['data'].append([ext_septuple] + features_per_packet)
				self.traffic_objects_to_dump['data'].append([ext_septuple] + self.dataset_per_traffic_object[ext_septuple])
	
	def serialize_all_matrix(self):
		'''
		Serializes MxN features extracted from each traffic_object in the .pickle file
		'''
		if not os.path.isdir('./global_PICKLEs/'):
			os.mkdir('./global_PICKLEs/')

		counter = 0
		counter_str = '000'
		new_out_pickle = './global_PICKLEs/' + self.out_pickle + '_' + counter_str

		while os.path.exists(new_out_pickle):
			counter += 1
			counter_str = str(counter)
			if counter < 100: counter_str = '0' + counter_str
			if counter < 10: counter_str = '0' + counter_str
			new_out_pickle = './global_PICKLEs/' + self.out_pickle + '_' + counter_str

		with open('%s_%s_TW_%s_packets.pickle' % (new_out_pickle, self.traffic_object_name, TIME_WINDOW), 'wb') as serialize_dataset:
			pickle.dump(self.packets_to_dump, serialize_dataset, pickle.HIGHEST_PROTOCOL)
			# pickle.dump(self.packet_samples, serialize_dataset, pickle.HIGHEST_PROTOCOL)
			# pickle.dump(self.packet_labels, serialize_dataset, pickle.HIGHEST_PROTOCOL)

		with open('%s_%s_TW_%s.pickle' % (new_out_pickle, self.traffic_object_name, TIME_WINDOW), 'wb') as serialize_dataset:
			pickle.dump(self.traffic_objects_to_dump, serialize_dataset, pickle.HIGHEST_PROTOCOL)
			# pickle.dump(self.traffic_object_samples, serialize_dataset, pickle.HIGHEST_PROTOCOL)
			# pickle.dump(self.traffic_object_labels, serialize_dataset, pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":

	available_datasets = {
		0: 'UNSWIOT',
		1: 'BOTIOT',
		2: 'CIC18'
	}

	available_traffic_objects = {
		0: 'BICHANNEL',					# Aggregate of packets per SRC/DST MACs/IPs, without consider PORTs
		1: 'BIFLOW',					# Aggregate of packets per SRC/DST MACs/IPs/PORTs
		2: 'CONNECTION',				# Aggregate of packets per SRC/DST MACs/IPs/PORTs, with TCP HEURISTIC and TIMEOUTS
		3: 'BAG_OF_SENT_PACKETS',		# Aggregate of packets per SRC MACs/IPs
		4: 'BAG_OF_RECEIVED_PACKETS'	# Aggregate of packets per DST IPs
	}

	dataset = 0
	traffic_object = 0

	if len(sys.argv) < 5:
		print('Usage:', sys.argv[0], '<LABELS_FILE>', '<PCAP_FILE>', '<OUT_PICKLE>', '<DATASET>', '<TRAFFIC OBJECT>')
		print('Available datasets', available_datasets)
		print('Available traffic objects', available_traffic_objects)
		sys.exit(1)

	label_file = sys.argv[1]
	pcap_file = sys.argv[2]
	out_pickle = sys.argv[3]
	if len(sys.argv) >= 4:
		dataset = int(sys.argv[4])
	if len(sys.argv) >= 5:
		traffic_object = int(sys.argv[5])

	if dataset not in available_datasets:
		print('Available datasets', available_datasets)
		sys.exit(1)
	dataset_name = available_datasets[dataset]
	traffic_object_name = available_traffic_objects[traffic_object]

	print('Analyzing:', label_file, pcap_file)
	t=time.time()
	traffic_object_extractor = TrafficObjectPickle(label_file, pcap_file, out_pickle, dataset_name, traffic_object_name)
	print(time.time() - t,'\n')
	print('Extracting labels info for dataset %s...' % dataset_name)
	t=time.time()
	traffic_object_extractor.factory_methods('d', 'retrieve_labels')
	print(time.time() - t,'\n')
	print('Extracting %s from PCAP...' % traffic_object_name)
	t=time.time()
	traffic_object_extractor.factory_methods('t', 'pcap_extraction')
	print(time.time() - t,'\n')
	print('Preparing dataset for serialization...')
	t=time.time()
	traffic_object_extractor.factory_methods('t', 'features_extraction_and_labelling', [ False ])
	traffic_object_extractor.prepare_dataset_for_dump()
	print(time.time() - t,'\n')
	print('Serializing PICKLE ouput...')
	t=time.time()
	traffic_object_extractor.serialize_all_matrix()
	print(time.time() - t,'\n')
