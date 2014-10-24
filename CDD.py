# CDD.py
from __future__ import division

#import scipy as sp
import numpy as np
import CDD_defaults as Cd

default_config = {
	'nx':2**6,			# Number of grid points in each dimension (can be a vector)
	'nt':2**9,			# Number of time steps
	'X':None,			# Total spatial size
	'T':None,			# Total time
	'dx':.01,			# Lattice spacing
	'dt':.01,			# Time step
	'dim':2,			# Spatial dimension
	'out':Cd.out0,		# Output function
	'stepper':Cd.RK,	# Time stepper function
	'method':Cd.CUW,	# Differencing method
	'flux':Cd.flux0,	# Flux function
	'adaptive':False,	# Adaptive time steps?
	'disordr':False,	# Disorder field?  Set this to the paramters of the disorder field (see below)
	'snapX':True,		# Snap X to dx*nx (otherwise snap dx to X/nx)
	'snapT':True,		# Snap T to dt*nt (otherwise snap dt to T/nt)
	'opfInit':(1.,.1),	# Initialization parameters for the order parameter field (see below)
	'seed':0			# Seed for random number generator
	}


def solve(nx=None,nt=None,X=None,T=None,dx=None,dt=None,dim=None,
			config=None,configFile=None,out=None,
			stepper=None,method=None,flux=None,
			adaptive=False,disordr=False,snapX=True,snapT=True,
			seed=None,opfInit=None,state0=None,configer=None,vb=True,**xtra):
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
		
		'''
	
	if configer==None:			# The configer is a function that sets the configuration
		configer = configer0
	
	# Set up config, state, and dyn:
	config = configer(nx,nt,X,T,dx,dt,dim,out,stepper,method,flux,\
		adaptive,disordr,snapX,snapT,seed,config,configFile,vb,xtra=xtra)

	state = stateInit(config,state0,opfInit,vb,xtra)
	# While initializing dyn, it may be good to update state as well, to
	# allow for the case where dyn will store previous values of state.
	# In such a case, dynInit might do preliminary time steps of state
	# to get the needed number of previous states in dyn, at which point
	# normal execution of time steps may start.
	#######(dyn,state) = dynInit(config,state,vb,xtra=xtra)
	
	return state,config
	
	#state,dyn = run(state,dyn,config,vb)

#---------------------------------------------------------------------
def configer0(nx=None,nt=None,X=None,T=None,dx=None,dt=None,dim=None,
	out=None,stepper=None,method=None,flux=None,adaptive=None,disordr=None,
	snapX=None,snapT=None,seed=None,config0=None,configFile=None,
	vb=None,**xtra):
	'''Sets up the configuration details for the CDD simulation. 
		The default setup comes from CDD_defaults, which is over-riden by
		the configuration file provided in 'configFile', which is 
		over-riden by the variables in the dictionary 'config0', 
		which is over-riden by explicit arguments. '''
	if vb: print "configer0(): Setting up configuration."
	
	# The way this works is that everything goes through 'config' while
	# we set up precedence, and then after that is done we split up 
	# 'config' into seperate 'state' and 'dyn'
	config = default_config
	if configFile is not None: config.update(readConfigFile(configFile))
	if config0 is not None: config.update(config0)
	for (k,v) in zip(['nx','nt','X','T','dx','dt','dim','out','stepper',\
			'method','flux','adaptive','disordr','snapX','snapT','seed','xtra'],\
			[nx,nt,X,T,dx,dt,dim,out,stepper,method,flux,adaptive,disordr,snapX,snapT,seed,xtra]):
		if v is not None:
			config.update({k:v})
	
	# Next make all grid parameters to be vectors (recall that scalar inputs are allowed):
	for k in ['nx','dx','X']:
		config[k] = vectorateParam(config[k],config['dim'])

	# Now check everything for consistency:
	return checkConfig(config,vb)

#---------------------------------------------------------------------
def vectorateParam(p,dim):
	'''Makes sure that parameter p is a vector of dimension dim.
		If p is a scalar, it gets replicated into a vector.  If it's a 
		vector of length != dim, then the first element gets replicated
		into a vector of length dim, and a warning is issued.'''
	if np.isscalar(p):
		p = np.array([p]*dim)	# This makes it a vector
	elif hasattr(p,'__len__') and p.shape[0] != dim and p.shape[0] == 1:
		p = np.array([p[0]]*dim)
	elif hasattr(p,'__len__'):
		raise ValueError("Warning: a grid parameter has a different dimension than the simulation:"+str(p))
	return p

#---------------------------------------------------------------------
def checkConfig(config,vb=None):
	'''Does error checking on 'config' to make sure that everything is reasonable.'''
	if vb: print "checkConfig(): Checking configuration..."
	
	# Check that two of nx, X, and dx are specified:
		# The sum[...] below counts how many of nx,dx,X are non-null.
		# If this is less than 2, there's nothing we can do but raise an error.
		# If this is greater than 2, we have an overlap, and we discard nx.
	
	s = sum([int(np.any(config[k])) for k in {'nx','dx','X'}])
	if s < 2:
		raise ValueError("Not enough grid parameters specified.")
	elif s > 2:
		config['nx'] = int(config['X']/config['dx'])
		raise UserWarning("Grid parameters overspecified.  Using: \n\t nx="\
			+str(config['nx'])+"\n\t X="+str(config['X'])+"\n\t dx="+\
			str(config['dx']))
	else:
		if config['nx']==None:
			config['nx'] = int(config['X']/config['dx'])
		elif config['dx']==None:
			config['dx'] = config['X']/config['nx']
		else:
			config['X']  = config['nx']*config['dx']
	
	# Now make the relationship between nx, X, and dx exact:
	if config['snapX']:
		config['X'] = config['nx']*config['dx']
	else:
		config['dx'] = config['X']/config['dx']
	
	# Repeat the above procedure for the time parameters:
	s = sum([int(np.any(config[k])) for k in {'nt','dt','T'}])
	if s < 2:
		raise ValueError("Not enough time parameters specified.")
	elif s > 2:
		config['nt'] = int(config['T']/config['dt'])
		raise UserWarning("Time parameters overspecified.  Using: \n\t nt="\
			+str(config['nt'])+"\n\t T="+str(config['T'])+"\n\t dt="+\
			str(config['dt']))
	else:
		if config['nt']==None:
			config['nt'] = int(config['T']/config['dt'])
		elif config['T']==None:
			config['T'] = config['nt']*config['dt']
		else:
			config['dt'] = config['T']/config['nt']
	
	# Make relationship between nt, T, and dt exact:
	if config['snapT']:
		config['T'] = config['nt']*config['dt']
	else:
		config['dt'] = config['T']/config['nt']
	
	return config

#---------------------------------------------------------------------
def stateInit(config,state0,opfInit,vb=None,xtra=None):
	'''Initializes the CDD state.  If state0 is provided, it is the initial
	state, and it can be specified in several ways.  If state0 is a member 
	of the CDDState class, it is taken to be precisely the initial state, 
	so output state=state0.  If state0 is a string, it is taken to be a
	file name from which the initial state is taken; in this case, the 
	extension determines how the file is processed.'''
	
	if isinstance(state0,CDDState): return state0	# This is the easy case.
	
	if state0==None: return CDDState(config)		# This is the other easy case.
	
	############ Fill this in later

#---------------------------------------------------------------------
class CDDState(dict):
	'''CDDState records the CDD state at a moment of time.
		The properties stored are grid shape, order parameter field,
		disorder field, a method for computing flux, and a seed used to 
		initialize the fields.'''
	
	keyz = {'nx','X','dx','dim','flux','disordr','seed'}
	def __init__(self, config):
		self.d = dict( [(k,config[k]) for k in self.keyz] )
		
		if self.d['seed']==0:
			self.d['opf'] = np.zeros( ([3,3])+list(self.d['nx']) )
	##################
