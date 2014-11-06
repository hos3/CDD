# CDD.py
from __future__ import division
from __future__ import print_function

#import scipy as sp
import numpy as np
import CDD_defaults as Cd

default_config = {
	'dim':2,				# Spatial dimension
	'nx':2**5,				# Number of grid points in each dimension (can be a vector)
	'nt':2**9,				# Number of time steps
	'dx':.01,				# Lattice spacing
	'dt':.01,				# Time step
	'X':None,				# Total spatial size
	'T':None,				# Total time
	'T0':0,					# Starting time
	'Tf':None,				# Stopping time
	'adaptive':False,		# Adaptive time steps?
	'nstepper':1,			# Number of steps in the time stepper (1 is Euler)
	'disordr':False,		# Disorder field?  Set this to the paramters of the disorder field (see below)
	'snapX':True,			# Snap X to dx*nx (otherwise snap dx to X/nx)
	'snapT':True,			# Snap T to dt*nt (otherwise snap dt to T/nt)
	'opf0':(20,1.,.1),		# Initialization parameters for the order parameter field (see below)
	'seed':0,				# Seed for random number generator
	# The following are functions:
	'run':runPy,			# Time evolver
	'opfInit':betaInit,		# Order parameter field initializer
	'dynInit':dynInit		# Dynamics object initializer
	'stepper':Cd.AEuler,	# Time stepper function
	'method':Cd.vv,			# Differencing method
	'flux':Cd.flux0,		# Flux function
	'out':Cd.out0			# Output function
	}


def solve(config=None, vb=True, **xtra):
	'''Solver for the CDD equations - under development.
		The 'vb' parameter stands for 'verbose' and indicates that the 
		program should output status updates while running.
		The program is organized around three structures, named 'state',
		'dyn' (short for 'dynamics'), and 'config' (short for 
		'configuration').  'state' is designed to hold all the informa-
		tion relevant for the state of the CDD system at an instant of 
		time, i.e. the grid size, order parameter fields, disorder 
		fields, flux function, etc.  It is designed to be usable as a 
		return variable, so that everything one could want to know about
		the system at a moment of time can be determined from it. 
		'dyn' is designed to hold all the information needed to evolve 
		the system in time, i.e. the time step size dt, the time step 
		method, and state variables from previous time steps, if needed. 
		'config' is a dictionary holding everything else, namely i/o 
		settings and such.  You can input ALL configurable settings 
		through 'config', including features that belong to state or
		dyn, like intial state ('state0'), 'stepper', etc.  Config also 
		contains some parameters duplicated in 'state' and 'dyn', like 
		grid shape and dt, for ease of access (the rule is that, if it's
		something that you might want to tweak on its own, like a scalar
		or small vector, it's in config; functions like the time stepper
		and big arrays like order parameter fields are left out).
		
		You can specify grid shape via any two of nx, X, or dx 
		(representing number of grid points, total grid extent, and 
		grid spacing, respectively).  Any of these can be a vector of 
		length 'dim' (the system's spatial dimension) or a scalar (in 
		which case it is automatically promoted to a vector as above). 
		The same story applies to time parameters nt, T, or dt.
		If you specify all three in either case, a warning is issued
		and the step size (dt or dx) and total extent (T or X) are used.
		
		(nx=None,nt=None,X=None,T=None,dx=None,dt=None,dim=None,
			config=None,configFile=None,out=None,
			stepper=None,method=None,flux=None,
			adaptive=False,disordr=False,snapX=True,snapT=True,
			seed=None,opfInit=None,state0=None,configer=None,vb=True,**xtra)
		
		'''
	
	# Polish up config so that it's ready to go for the simulation:
	config = configer(config,xtra,vb=vb)	# config has all configuration details
	
	# Initialize the state object:
	state = stateInit(config,vb=vb)
	
	''' While initializing dyn, it may be good to update state as well, to
		allow for the case where dyn will store previous values of state.
		In such a case, dynInit might do preliminary time steps of state
		to get the needed number of previous states in dyn, at which point
		normal execution of time steps may start. '''
	#dyn,state,t,dt = config['dynInit'](config,state,vb)

	return state,config
	
	#state,t,dt,dyn = config['run'](state,dyn,vb)
	

#---------------------------------------------------------------------
def runPy(state,dyn,vb=True):
	'''Evolves the state, using Python scripting (i.e. no Cuda).
		
		REQUIRES dyn TO HAVE: 
			t		- Current time
			Tf		- Final time
			dt		- Time step
			
	'''
	
	t,Tf,dt = dyn.t,dyn.Tf,dyn.dt
	while t<Tf:
		state,t,dt = dyn.stepper(state,t,Tf,dt)

#---------------------------------------------------------------------
def configer(config,xtra,vb=True):
	'''Sets up the configuration details for the CDD simulation. 
		The default setup comes from CDD_defaults, which is over-riden by
		the configuration file provided in 'configFile', which is 
		over-riden by the variables in the dictionary 'config0', 
		which is over-riden by explicit arguments. '''
	
	if vb: print("configer0(): Setting up configuration.")
	
	# The way this works is that everything goes through 'config' while
	# we set up precedence, and then after that is done we split up 
	# 'config' into seperate 'state' and 'dyn'
	
	if config is None: config = {}
	config.update(xtra)			# First update config with "command line" arguments
	
	try:						# Next look for a configuration file
		configFile = config['configFile']	# User-supplied file
	except KeyError:
		try:
			configFile = default_config['configFile']	# Default file
		except KeyError:					# No config file
			if vb: print("No config file provided")
			configFile = None
	
	# Now start building the final config dict:
	config0 = default_config.copy()
	if configFile is not None: config0.update(readConfigFile(configFile))
	config0.update(config)
	config = config0		# Copy config0 back to config
	
	# Allow overriding of arbitrary global defaults: (e.g. if you want to change 'checkConfig' or something)
	if 'globl' in config.keys():
		globals().update(config.globl)
	
	# Next make all grid parameters to be vectors (recall that scalar inputs are allowed):
	if vb: print("Vectorizing grid parameters...")
	for k in ['nx','dx','X']:
		config[k] = vectorateParam(config[k],config['dim'])
		if vb: print(k+" = "+str(config[k]))
	
	# Seed the random number generator:
	if 'seed' in config.keys():
		np.random.seed(config['seed'])
	
	# Now check everything for consistency:
	return checkConfig(config,vb)

#---------------------------------------------------------------------
def vectorateParam(p,dim):
	'''Makes sure that parameter p is a vector of dimension dim.
		If p is a scalar or length 1 iterable, it gets replicated 
		into a vector.  If p is a length dim iterable, it is made into
		a numpy vector.  If p is None, then it just gets returned.  
		Otherwise a ValueError is raised.'''
	
	# Case 0: If p has length == dim, return p as a numpy vector:
	if hasattr(p,'__len__') and p.shape[0] == dim:
		p = np.array([i for i in p])
		
		# Case 1: if p is a genuine scalar, make it a vector:
	elif np.isscalar(p):
		p = np.array([p]*dim)
		
		# Case 2: If p is a length 1 array, make it longer:
	elif hasattr(p,'__len__') and p.shape[0] != dim and p.shape[0] == 1:
		p = np.array([p[0]]*dim)
		
		# Case 3: If p is None, return None:
	elif p is None:
		pass
		
	else:
		raise ValueError("Warning: a grid parameter has a different dimension than the simulation:"+str(p))
		
	return p

#---------------------------------------------------------------------
def checkConfig(config,vb=True):
	'''Does error checking on 'config' to make sure that everything is reasonable.'''
	if vb: print("checkConfig(): Checking configuration...")
	
	# Check that two of nx, X, and dx are specified:
		# The sum[...] below counts how many of nx,dx,X are non-null.
		# If this is less than 2, there's nothing we can do but raise an error.
		# If this is greater than 2, we have an overlap, and we discard nx.
	
	s = sum([int(np.any(config[k])) for k in {'nx','dx','X'}])
	if s < 2:
		raise ValueError("Not enough grid parameters specified.")
	elif s > 2 and (config['X']!=config['dx']*config['nx']):
		config['nx'] = int(config['X']/config['dx'])
		print("Warning: Grid parameters overspecified.  Using: \n\t nx="\
			+str(config['nx'])+"\n\t X="+str(config['X'])+"\n\t dx="+\
			str(config['dx']))
	else:
		if config['nx']==None:
			config['nx'] = int(config['X']/config['dx'])
			if vb: print('Setting nx = '+srt(config['nx']))
		elif config['dx']==None:
			config['dx'] = config['X']/config['nx']
			if vb: print('Setting dx = '+str(config['dx']))
		else:
			config['X']  = config['nx']*config['dx']
			if vb: print('Setting X = '+str(config['X']))
	
	# Now make the relationship between nx, X, and dx exact:
	if config['snapX']:
		config['X'] = config['nx']*config['dx']
	else:
		config['dx'] = config['X']/config['dx']
	
	# If starting and ending times are specified, fill in for total time:
	if config['T0'] is not None and config['Tf'] is not None:
		if config['T'] is not None:
			print("Warning: Time range overspecified. Using: T="+str(config['Tf']-config['T0']))
		config['T'] = config['Tf']-config['T0']
	
	# Do an alignment of the time parameters (as for the grid parameters):
	s = sum([int(np.any(config[k])) for k in {'nt','dt','T'}])
	if s < 2:
		print("Warning: Not enough time parameters specified.")
	elif s > 2 and (config['T']!=config['dt']*config['nt']):
		config['nt'] = int(config['T']/config['dt'])
		print("Warning: Time parameters overspecified.  Using: \n\t nt="\
			+str(config['nt'])+"\n\t T="+str(config['T'])+"\n\t dt="+\
			str(config['dt']))
	else:
		if config['nt']==None:
			config['nt'] = int(config['T']/config['dt'])
			if vb: 'Setting nt = '+str(config['nt'])
		elif config['T']==None:
			config['T'] = config['nt']*config['dt']
			if vb: 'Setting T = '+str(config['T'])
		else:
			config['dt'] = config['T']/config['nt']
			if vb: 'Setting dt = '+str(config['dt'])
	
	# Make relationship between nt, T, and dt exact:
	if config['snapT']:
		config['T'] = config['nt']*config['dt']
	else:
		config['dt'] = config['T']/config['nt']
	
	# Re-align time interval bounds:
	if config['T0'] is None and config['Tf'] is None:
		print('Neither time interval endpoint is set.  Using T0=0')
		config['T0'] = 0
		config['Tf'] = config['T']
	elif config['T0'] is not None:
		if config['Tf'] != config['T']+config['T0']: 
			print('Warning: Tf has changed.  New value: '+str(config['T']+config['T0']))
		config['Tf'] = config['T']+config['T0']
	else:
		config['T0'] = config['Tf']-config['T']
	
	# Set current time t to starting time T0:
	config['t'] = T0
	
	if vb: print("Config check done.")
	return config

#---------------------------------------------------------------------
def stateInit(config,vb=True):
	'''Initializes the CDD state.  If state0 is provided, it is the initial
		state, and it can be specified in several ways.  If state0 is a member 
		of the CDDState class, it is taken to be precisely the initial state, 
		so output state=state0.  If state0 is a string, it is taken to be a
		file name from which the initial state is taken; in this case, the 
		extension determines how the file is processed.
		
		REQUIRES CONFIG TO HAVE: (no requirements)
		'''
	if vb: print("Initializing state...")
	
	if (not 'state0' in config.keys()) or (cofig['state0'] is None): # Easy case: just make new object
		if vb: print("No initial data provided.  Initializing new state from seed.")
		return CDDState(config)
	else:
		if isinstance(config['state0'],CDDState):	# Another easy case: Use supplied object
			if vb: print("Using provided initial state.")
			return state0
		
		############ Fill this in later

#---------------------------------------------------------------------
class CDDState():
	'''CDDState records the CDD state at a moment of time.
		The properties stored are grid shape, order parameter field,
		disorder field, a method for computing flux, and a seed used to 
		initialize the fields.
		
		REQUIRES CONFIG TO HAVE:
			nx		-Number of grid points in each dimension
			X		-Total spatial extent in each dimension
			dx		-Grid spacing in each dimension
			dim		-Number of dimensions
			flux	-Flux function
			disordr	-Is there disorder or not
		'''
	
	attr = {'nx','X','dx','dim','flux','disordr'}
	def __init__(self, config,vb=True):
		
		if vb: print("Initializing CDDState object...")
		for a in self.attr:
			setattr(self,a,config[a])
		
		if vb: print("Initializing order parameter field...")
		self.opf = config['opfInit'](config,vb)
		
		if config['disordr']:
			if vb: print("Initializing disorder field...")
			N,A,w = config['disordr']
			self.disordr = randGssnField(self.dx,self.nx,N,A,w)
		
		if vb: print("Done initializing CDDState object.")
	
	def __getitem__(self,slc):
		'''Fancy indexing routine.  The way this works is as follows:
			'slc' represents an index or slice or tuple of slices
			(i.e. what you get by putting '[stuff]' after the object name).
			If the first (or only) element of slc is an attribute name,
			then the return value comes from that attribute.  If the first
			element of slc doesn't match any attribute name, it is assumed
			that the user is looking for the order parameter field.
			All the elements of slc that remain are then passed through a 
			conversion function (sliceByXYZ), which allows indexing by 
			letters like 'x', 'y', and 'z', and then the resulting 
			reformatted slice is passed to the attribute in question.
			'''
		gattr = 'opf'		# Get this attribute; by default, look at the order parameter field beta
		
		if not isinstance(slc,tuple):	# If user calls self[stuff] and stuff has no commas
			if slc in self.attr:
				return getattr(self,slc)
			else:
				return getattr(self,gattr)[sliceByXYZ(slc)]
			
		else:
			if isinstance(slc[0],str) and slc[0] in dir(self):
				return getattr(self,slc[0])[tuple(map(sliceByXYZ,slc[1:]))]
			else:
				return getattr(self,gattr)[tuple(map(sliceByXYZ,slc))]	

#---------------------------------------------------------------------
def sliceByXYZ(slc,root='x'):
	'''Takes a slice slc and looks to see whether it has the form
		slice('x','y',1) or something similar; if it does, then all the 
		chars are replaced by corresponding integers, via the map
		'x' -> 0, 'y' -> 1, 'z' -> 2.  ALSO, THE STOP INDEX IS 
		INCREMENTED BY 1 TO MAKE THE BOUND INCLUSIVE.  THIS IS NEEDED 
		SO THAT 'z' CAN BE ACCESSED EASILY.
		
		IF THE UPPER BOUND OF slc IS INPUT AS A STRING, THEN IT WILL BE INCLUSIVE!
		
		The mapping of 'x' to 0 can be modified by changing the optional
		parameter 'root' to something other than 'x'.
		Several alternative forms are accepted for specifying the slice:
		If slc is a single char, it is converted to a single int.
		If slc has the form 'x:z:2' it is converted to a slice of the 
			form 0:3:2.  (NOTE UPPER BOUND IS INCLUSIVE)
		If slc has the form 'xz' it is converted to a slice of the form
			0:3.  (NOTE UPPER BOUND IS INCLUSIVE)
		'''
	root = root.lower()			# For consistency, everything will be lowercase.
	def c2i(a):					# This turns a char index into an int index (with 'x' -> 0)
		if a.isdigit():			# If this is a string of a number, return the number
			return int(a)
		elif len(a)==1:			# If this is a char, return its ASCII ordinal (minus root ordinal)
			return ord(a)-ord(root)
		elif len(a)==0:			# If this is an empty string, return None (this works with slices)
			return None
		else:					# Otherwise, there's a problem
			raise ValueError("Function sliceByXYZ says: Ack! Too many letters in slice "+str(slc))
	
	if isinstance(slc,int):		# If we have an integer, just return
		return slc
	elif isinstance(slc,slice):	# If we have a slice object, just format the start, stop and step to be integers
		start, stop, step = slc.start, slc.stop, slc.step
		if isinstance(start,str): start = c2i(start)		# This makes start an integer
		if isinstance(stop,str): stop = c2i(stop)+1			# NOTE INCLUSIVE UPPER BOUND
		if isinstance(step,str): step = c2i(step)
		return slice(start,stop,step)
		
		# If we have a string, there are several possible interpretations:
	elif isinstance(slc,str):
		# The case with colons is the hardest, requiring several sub-cases:
		if ':' in slc:
			slc = map(c2i,slc.split(':'))	# First break up stuff between colons
			if len(slc)==3: 				# If three items supplied, then they should be start:stop:step
				slc = slice(slc[0],slc[1]+1,slc[2])			# NOTE INCLUSIVE UPPER BOUND
			elif len(slc)==2:				# If two items supplied, then they should be start:stop
				slc = slice(slc[0],slc[1]+1)				# NOTE INCLUSIVE UPPER BOUND
			else:
				raise ValueError("Function sliceByXYZ says: Ack! Too many colons in slice "+slc)
			
			# The remaining cases are easier:
		elif len(slc)==1: 					# If it's a single char, just change to int
			slc = c2i(slc)
		elif len(slc)==2:					# If it is just two letters, make an inclusive slice
			slc = slice(c2i(slc[0]),c2i(slc[1])+1)			# NOTE INCLUSIVE UPPER BOUND
		else:
			raise ValueError("Function sliceByXYZ says: Ack! Too many leters in slice "+slc)
		
	else:
		raise ValueError("Function sliceByXYZ doesn't like your slice "+str(slc))
	
	return slc

#---------------------------------------------------------------------
def betaInit(config, vb=True):
	'''Initializes an order parameter field (opf), either using a provided
		opf or else making a random new one from a tuple of parameters
		(N,A,w), where N is the number of Gaussian peaks to make, A is 
		their amplitude, and w is their width.
		
		REQUIRES CONFIG TO HAVE:
			dx		-Grid spacing
			nx		-Grid shape
			opf		-Data to create the initial opf; should either be a 
						full opf or else a tuple of the form (N,A,w)
		'''
	if vb: print("Initializing beta field...")
	
	# Easy case first: if user gives an opf, use that:
	if hasattr(config['opf0'],'shape') and config['opf0'].shape==(3,3)+(config['nx'],)*dim:
		beta = config['opf0']
		
		# Otherwise, if config is a tuple:
	elif isinstance(config['opf0'],(0,).__class__):
		'''In this case, we take opf0 to have the form (N,A,w), where
			N is a number of random Gaussian peaks to produce, A is their
			amplitude, and w is their width.'''
		dx = config['dx']
		nx = config['nx']
		N,A,w = config['opf0']
		
		beta = np.zeros([3,3]+[i for i in nx])
		for i in range(3):
			for j in range(3):
				beta[i,j,...] = randGssnField(dx,nx,N,A,w)
	
	return beta

#---------------------------------------------------------------------
def randGssnField(dx,nx,N,A,w,mode='wrap'):
	'''Makes a field of N randomly located Gaussians, each with height A
		and width w. 
		dx gives the lattice spacing 
		nx the number of lattice points (dx and nx should be vectors of the same length).
		N is the number of Gaussians to make
		A is the amplitude of the Gaussians
		w is the width of the Gaussians
		'''
	from scipy.ndimage import gaussian_filter as gfilter
	
	w = w/np.array([i for i in dx])		# Format w for input to gfilter
	
	rgf = np.zeros(nx)		# This will hold the random Gaussian field
	# Get coordinates for the centers of the Gaussians:
	idxs = np.floor( np.random.rand(len(nx),N) * np.array([nx]).T ).astype(int)
	# Set grid locations specified by above coordinates to the desired amplitude:
	rgf[ [idxs[i] for i in range(len(nx))] ] = A
	# Apply Gaussian filter:
	rgf = gfilter(rgf,sigma=w,mode=mode)
	
	return rgf

#---------------------------------------------------------------------
def dynInit(config,state,vb=True):
	'''Initializes the dynamics for the CDD state. 
		
		REQUIRES CONFIG TO HAVE:
			dt		- Time step
			t		- Current time
			stepper	- Time stepper
			method	- 
			
		'''
	from inspect import getargspec
	"""The first job is to find out how many steps the stepper is.
	 If the user provides this, then we're set.  Otherwise, we're going
	 to guess that it is one less than the number of required arguments
	 of the stepper (the other argument being the differencing method)."""
	if 'nsteps' in config.keys():
		nsteps = config['nsteps']
	else:
		nsteps,temp,temp,deflt = getargspec(config['stepper'])
		nsteps = len(nsteps)-len(deflt)-1
	
	"""If nsteps > 1, then we need to evolve the first few time steps 
	 differently before turning things over to the normal stepper. 
	 By default, we'll use an Euler stepper to do this initial evolving."""
	if nsteps > 1:
		prev = [state]		# prev will be a list of previous states needed for the stepper
		for i in range(nsteps-1):
			
	
