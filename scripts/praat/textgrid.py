# -*- coding: utf-8 -*-
import sys,codecs

from .intervaltier import *
from .pointtier import *
from .point import *
from .interval import *
from .textgridtype import *
from .tiertype import *
from fileinput import close

class Textgrid:	
	def __init__(self):
		self.file = ''
		self.btime = 0.0
		self.etime = 0.0
		self.nr_tiers = 0
		self.tiers = []
		self.type = TextgridType.LONG
		
	def read(self, file):
		self.file = file
		lines = self.read_file(self.file)
		self.determine_type(lines)
		self.read_header(lines)
		self.read_tiers(lines)
	
	def read_file(self,file):
		file_handle = codecs.open(file, 'r','utf-8')
		lines = file_handle.readlines()
		lines = [line.strip() for line in lines]
		file_handle.close()
		return lines
	
	def determine_type(self, lines):
		if lines[3].startswith('xmin'):
			self.type = TextgridType.LONG
		else:
			self.type = TextgridType.SHORT
	
	def read_header(self, lines):
		if self.type == TextgridType.LONG:
			self.btime = float(lines[3].split('=')[1].strip())
			self.etime = float(lines[4].split('=')[1].strip())
			self.nr_tiers = int(lines[6].split('=')[1].strip())
		elif self.type == TextgridType.SHORT:
			self.btime = float(lines[3])
			self.etime = float(lines[4])
			self.nr_tiers = int(lines[6])
	
	def read_tiers(self, lines):
		tier_index = 0
		if self.type == TextgridType.LONG:
			line_index = 8
			while tier_index < self.nr_tiers:
				last_tier_line = self.read_tier(lines, line_index)
				line_index = last_tier_line + 1
				tier_index += 1
		
		elif self.type == TextgridType.SHORT:
			line_index = 7
			while tier_index < self.nr_tiers:
				last_tier_line = self.read_tier(lines, line_index)
				line_index = last_tier_line + 1
				tier_index += 1
	
	def read_tier(self, lines, starting_line_index):
		if self.type == TextgridType.SHORT:
			type = lines[starting_line_index].strip()[1:-1]
			name = lines[starting_line_index+1].strip()[1:-1]
			btime = float(lines[starting_line_index+2])
			etime = float(lines[starting_line_index+3])
			
			if type == TierType.INTERVAL:
				nr_intervals = int(lines[starting_line_index+4])
				line_index = starting_line_index+4
				interval_index = 0
				intervals = []
				
				while interval_index < nr_intervals:
					interval_btime = float(lines[line_index+1])
					interval_etime = float(lines[line_index+2])
					text = lines[line_index+3].strip()[1:-1]
					interval = Interval(interval_btime, interval_etime, text)
					intervals.append(interval)
					
					line_index+=3
					interval_index += 1
				
				tier = IntervalTier(name, btime, etime, nr_intervals, intervals)
				self.tiers.append(tier)
				
				return line_index
				
			elif type == TierType.POINT:
				nr_points = int(lines[starting_line_index+4])
				line_index = starting_line_index+4
				point_index = 0
				points = []
				
				while point_index < nr_points:
					time = float(lines[line_index+1])
					text = lines[line_index+2].strip()[1:-1]
					point = Point(time, text)
					points.append(point)
					
					line_index+=2
					point_index += 1
				
				tier = PointTier(name, btime, etime, nr_points, points)
				self.tiers.append(tier)
				
				return line_index
		
		elif self.type == TextgridType.LONG:
			type = lines[starting_line_index+1].split('=')[1].strip()[1:-1]
			name = lines[starting_line_index+2].split('=')[1].strip()[1:-1]
			btime = float(lines[starting_line_index+3].split('=')[1])
			etime = float(lines[starting_line_index+4].split('=')[1])
			
			if type == TierType.INTERVAL:
				nr_intervals = int(lines[starting_line_index+5].split('=')[1])
				line_index = starting_line_index+6
				interval_index = 0
				intervals = []
				
				while interval_index < nr_intervals:
					btime = float(lines[line_index+1].split('=')[1])
					etime = float(lines[line_index+2].split('=')[1])
					text = lines[line_index+3].split('=')[1].strip()[1:-1]
					interval = Interval(btime, etime, text)
					intervals.append(interval)
					
					line_index+=4
					interval_index += 1
				
				tier = IntervalTier(name, btime, etime, nr_intervals, intervals)
				self.tiers.append(tier)
				
				return line_index-1
				
			elif type == TierType.POINT:
				nr_points = int(lines[starting_line_index+5].split('=')[1])
				line_index = starting_line_index+6
				point_index = 0
				points = []
				
				while point_index < nr_points:
					time = float(lines[line_index+1].split('=')[1])
					text = lines[line_index+2].split('=')[1].strip()[1:-1]
					point = Point(time, text)
					points.append(point)
					
					line_index+=3
					point_index += 1
				
				tier = PointTier(name, btime, etime, nr_points, points)
				self.tiers.append(tier)
				
				return line_index-1
	
	def write(self, file=None): 
		if file:
			handle = codecs.open(file, 'w')
		else:
			handle = sys.stdout
			
		handle.write("File type = \"ooTextFile\"\n" + \
						"Object class = \"TextGrid\"\n" + \
						"\n" + \
						'xmin = ' + repr(self.btime) + "\n" + \
						'xmax = ' + repr(self.etime) + "\n" + \
						'tiers? <exists>' + "\n" + \
						'size = ' + repr(len(self.tiers)) + "\n" + \
						'item []:\n')
		
		tiers = self.tiers
		tier_index = 1
		for tier in tiers:
			handle.write('\titem [' + repr(tier_index) + ']:\n')

			if isinstance(tier, IntervalTier):
				handle.write('\t\tclass = \"IntervalTier\"\n' + \
							'\t\tname = \"' + tier.name + '\"\n' + \
							'\t\txmin = ' + repr(tier.btime) + '\n' + \
							'\t\txmax = ' + repr(tier.etime) + '\n' + \
							'\t\tintervals: size = ' + repr(len(tier.intervals)) + '\n')
				intervals = tier.intervals
				interval_index = 1
				
				for interval in intervals:
					handle.write('\t\tintervals [' + repr(interval_index) + ']\n' + \
								'\t\t\txmin = ' + repr(interval.btime) + "\n" + \
								'\t\t\txmax = ' + repr(interval.etime) + "\n" + \
								'\t\t\ttext = \"' + interval.text + '\"\n')
					interval_index += 1
			
			elif isinstance(tier, PointTier):
				handle.write('\t\tclass = \"TextTier\"\n' + \
							'\t\tname = \"' + tier.name + '\"\n' + \
							'\t\txmin = ' + repr(tier.btime) + "\n" + \
							'\t\txmax = ' + repr(tier.etime) + "\n" + \
							'\t\tpoints: size = ' + repr(len(tier.points)) + "\n")
				points = tier.points
				point_index = 1
				
				for point in points:
					handle.write('\t\tpoints [' + repr(point_index) + ']\n' + \
								'\t\t\ttime = ' + repr(point.time) + "\n" + \
								'\t\t\tmark = \"' + point.text + '\"\n')
					point_index += 1
			
			tier_index += 1
		if file:
			handle.close()

	""" Get the index of the first occurrence of an IntervalTier object having the specified 
	name.
	
	@param tier_name: Name of the tier to find the index for.
	@return: The index of the first occurrence of an IntervalTier object having the specified
	name or -1 if no such tier was found.
	""" 
	def get_tier_index_by_name(self, tier_name):
		tier_index = -1
		has_found = False
		i = 0;
		while not(has_found) and i<len(self.tiers):
			if self.tiers[i].name == tier_name:
				tier_index = i
				has_found = True
			i += 1
		
		return tier_index

	""" Get the first occurrence of the tier object that has the name ending with the specified
	string.

	@param name: The string the tier name has to end with.
	@return: The first occurrence of a tier object that has a name ending with the specified string.
	"""
	def get_tier_by_name(self, name):
		found_tier = None
		for tier in self.tiers:
			if tier.name.endswith(name):
				found_tier = tier
				break

		return found_tier
	
