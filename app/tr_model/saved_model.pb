??7
??
D
AddV2
x"T
y"T
z"T"
Ttype:
2	??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
>
Maximum
x"T
y"T
z"T"
Ttype:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
>
Minimum
x"T
y"T
z"T"
Ttype:
2	
?
Mul
x"T
y"T
z"T"
Ttype:
2	?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ??
@
StaticRegexFullMatch	
input

output
"
patternstring
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
-
Tanh
x"T
y"T"
Ttype:

2
?
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type*
output_handle??element_dtype"
element_dtypetype"

shape_typetype:
2	
?
TensorListReserve
element_shape"
shape_type
num_elements#
handle??element_dtype"
element_dtypetype"

shape_typetype:
2	
?
TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsint?????????
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?
?
While

input2T
output2T"
T
list(type)("
condfunc"
bodyfunc" 
output_shapeslist(shape)
 "
parallel_iterationsint
?"serve*2.7.02unknown8??5
z
dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*
shared_namedense_2/kernel
s
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel* 
_output_shapes
:
??*
dtype0
q
dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_2/bias
j
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes	
:?*
dtype0
y
dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*
shared_namedense_3/kernel
r
"dense_3/kernel/Read/ReadVariableOpReadVariableOpdense_3/kernel*
_output_shapes
:	?*
dtype0
p
dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_3/bias
i
 dense_3/bias/Read/ReadVariableOpReadVariableOpdense_3/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
?
lstm_2/lstm_cell_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?**
shared_namelstm_2/lstm_cell_2/kernel
?
-lstm_2/lstm_cell_2/kernel/Read/ReadVariableOpReadVariableOplstm_2/lstm_cell_2/kernel*
_output_shapes
:	?*
dtype0
?
#lstm_2/lstm_cell_2/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*4
shared_name%#lstm_2/lstm_cell_2/recurrent_kernel
?
7lstm_2/lstm_cell_2/recurrent_kernel/Read/ReadVariableOpReadVariableOp#lstm_2/lstm_cell_2/recurrent_kernel* 
_output_shapes
:
??*
dtype0
?
lstm_2/lstm_cell_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*(
shared_namelstm_2/lstm_cell_2/bias
?
+lstm_2/lstm_cell_2/bias/Read/ReadVariableOpReadVariableOplstm_2/lstm_cell_2/bias*
_output_shapes	
:?*
dtype0
?
lstm_3/lstm_cell_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??**
shared_namelstm_3/lstm_cell_3/kernel
?
-lstm_3/lstm_cell_3/kernel/Read/ReadVariableOpReadVariableOplstm_3/lstm_cell_3/kernel* 
_output_shapes
:
??*
dtype0
?
#lstm_3/lstm_cell_3/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*4
shared_name%#lstm_3/lstm_cell_3/recurrent_kernel
?
7lstm_3/lstm_cell_3/recurrent_kernel/Read/ReadVariableOpReadVariableOp#lstm_3/lstm_cell_3/recurrent_kernel* 
_output_shapes
:
??*
dtype0
?
lstm_3/lstm_cell_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*(
shared_namelstm_3/lstm_cell_3/bias
?
+lstm_3/lstm_cell_3/bias/Read/ReadVariableOpReadVariableOplstm_3/lstm_cell_3/bias*
_output_shapes	
:?*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
?
Adam/dense_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*&
shared_nameAdam/dense_2/kernel/m
?
)Adam/dense_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_2/kernel/m* 
_output_shapes
:
??*
dtype0

Adam/dense_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*$
shared_nameAdam/dense_2/bias/m
x
'Adam/dense_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_2/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/dense_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*&
shared_nameAdam/dense_3/kernel/m
?
)Adam/dense_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_3/kernel/m*
_output_shapes
:	?*
dtype0
~
Adam/dense_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_3/bias/m
w
'Adam/dense_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_3/bias/m*
_output_shapes
:*
dtype0
?
 Adam/lstm_2/lstm_cell_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*1
shared_name" Adam/lstm_2/lstm_cell_2/kernel/m
?
4Adam/lstm_2/lstm_cell_2/kernel/m/Read/ReadVariableOpReadVariableOp Adam/lstm_2/lstm_cell_2/kernel/m*
_output_shapes
:	?*
dtype0
?
*Adam/lstm_2/lstm_cell_2/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*;
shared_name,*Adam/lstm_2/lstm_cell_2/recurrent_kernel/m
?
>Adam/lstm_2/lstm_cell_2/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp*Adam/lstm_2/lstm_cell_2/recurrent_kernel/m* 
_output_shapes
:
??*
dtype0
?
Adam/lstm_2/lstm_cell_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*/
shared_name Adam/lstm_2/lstm_cell_2/bias/m
?
2Adam/lstm_2/lstm_cell_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/lstm_2/lstm_cell_2/bias/m*
_output_shapes	
:?*
dtype0
?
 Adam/lstm_3/lstm_cell_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*1
shared_name" Adam/lstm_3/lstm_cell_3/kernel/m
?
4Adam/lstm_3/lstm_cell_3/kernel/m/Read/ReadVariableOpReadVariableOp Adam/lstm_3/lstm_cell_3/kernel/m* 
_output_shapes
:
??*
dtype0
?
*Adam/lstm_3/lstm_cell_3/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*;
shared_name,*Adam/lstm_3/lstm_cell_3/recurrent_kernel/m
?
>Adam/lstm_3/lstm_cell_3/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp*Adam/lstm_3/lstm_cell_3/recurrent_kernel/m* 
_output_shapes
:
??*
dtype0
?
Adam/lstm_3/lstm_cell_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*/
shared_name Adam/lstm_3/lstm_cell_3/bias/m
?
2Adam/lstm_3/lstm_cell_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/lstm_3/lstm_cell_3/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/dense_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*&
shared_nameAdam/dense_2/kernel/v
?
)Adam/dense_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_2/kernel/v* 
_output_shapes
:
??*
dtype0

Adam/dense_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*$
shared_nameAdam/dense_2/bias/v
x
'Adam/dense_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_2/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/dense_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*&
shared_nameAdam/dense_3/kernel/v
?
)Adam/dense_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_3/kernel/v*
_output_shapes
:	?*
dtype0
~
Adam/dense_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_3/bias/v
w
'Adam/dense_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_3/bias/v*
_output_shapes
:*
dtype0
?
 Adam/lstm_2/lstm_cell_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*1
shared_name" Adam/lstm_2/lstm_cell_2/kernel/v
?
4Adam/lstm_2/lstm_cell_2/kernel/v/Read/ReadVariableOpReadVariableOp Adam/lstm_2/lstm_cell_2/kernel/v*
_output_shapes
:	?*
dtype0
?
*Adam/lstm_2/lstm_cell_2/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*;
shared_name,*Adam/lstm_2/lstm_cell_2/recurrent_kernel/v
?
>Adam/lstm_2/lstm_cell_2/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp*Adam/lstm_2/lstm_cell_2/recurrent_kernel/v* 
_output_shapes
:
??*
dtype0
?
Adam/lstm_2/lstm_cell_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*/
shared_name Adam/lstm_2/lstm_cell_2/bias/v
?
2Adam/lstm_2/lstm_cell_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/lstm_2/lstm_cell_2/bias/v*
_output_shapes	
:?*
dtype0
?
 Adam/lstm_3/lstm_cell_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*1
shared_name" Adam/lstm_3/lstm_cell_3/kernel/v
?
4Adam/lstm_3/lstm_cell_3/kernel/v/Read/ReadVariableOpReadVariableOp Adam/lstm_3/lstm_cell_3/kernel/v* 
_output_shapes
:
??*
dtype0
?
*Adam/lstm_3/lstm_cell_3/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*;
shared_name,*Adam/lstm_3/lstm_cell_3/recurrent_kernel/v
?
>Adam/lstm_3/lstm_cell_3/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp*Adam/lstm_3/lstm_cell_3/recurrent_kernel/v* 
_output_shapes
:
??*
dtype0
?
Adam/lstm_3/lstm_cell_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*/
shared_name Adam/lstm_3/lstm_cell_3/bias/v
?
2Adam/lstm_3/lstm_cell_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/lstm_3/lstm_cell_3/bias/v*
_output_shapes	
:?*
dtype0

NoOpNoOp
?:
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?9
value?9B?9 B?9
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
	optimizer
regularization_losses
	variables
trainable_variables
		keras_api


signatures
l
cell

state_spec
regularization_losses
	variables
trainable_variables
	keras_api
l
cell

state_spec
regularization_losses
	variables
trainable_variables
	keras_api
h

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
h

kernel
bias
regularization_losses
 	variables
!trainable_variables
"	keras_api
?
#iter

$beta_1

%beta_2
	&decay
'learning_ratemhmimjmk(ml)mm*mn+mo,mp-mqvrvsvtvu(vv)vw*vx+vy,vz-v{
 
F
(0
)1
*2
+3
,4
-5
6
7
8
9
F
(0
)1
*2
+3
,4
-5
6
7
8
9
?
regularization_losses
.non_trainable_variables
	variables
/layer_regularization_losses
trainable_variables
0metrics
1layer_metrics

2layers
 
?
3
state_size

(kernel
)recurrent_kernel
*bias
4regularization_losses
5	variables
6trainable_variables
7	keras_api
 
 

(0
)1
*2

(0
)1
*2
?
regularization_losses
8non_trainable_variables
	variables
9layer_regularization_losses
trainable_variables
:metrics
;layer_metrics

<states

=layers
?
>
state_size

+kernel
,recurrent_kernel
-bias
?regularization_losses
@	variables
Atrainable_variables
B	keras_api
 
 

+0
,1
-2

+0
,1
-2
?
regularization_losses
Cnon_trainable_variables
	variables
Dlayer_regularization_losses
trainable_variables
Emetrics
Flayer_metrics

Gstates

Hlayers
ZX
VARIABLE_VALUEdense_2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
?
regularization_losses
Inon_trainable_variables
	variables
Jlayer_regularization_losses
trainable_variables
Kmetrics
Llayer_metrics

Mlayers
ZX
VARIABLE_VALUEdense_3/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_3/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
?
regularization_losses
Nnon_trainable_variables
 	variables
Olayer_regularization_losses
!trainable_variables
Pmetrics
Qlayer_metrics

Rlayers
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUElstm_2/lstm_cell_2/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUE#lstm_2/lstm_cell_2/recurrent_kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUElstm_2/lstm_cell_2/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUElstm_3/lstm_cell_3/kernel&variables/3/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUE#lstm_3/lstm_cell_3/recurrent_kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUElstm_3/lstm_cell_3/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE
 
 

S0
T1
 

0
1
2
3
 
 

(0
)1
*2

(0
)1
*2
?
4regularization_losses
Unon_trainable_variables
5	variables
Vlayer_regularization_losses
6trainable_variables
Wmetrics
Xlayer_metrics

Ylayers
 
 
 
 
 

0
 
 

+0
,1
-2

+0
,1
-2
?
?regularization_losses
Znon_trainable_variables
@	variables
[layer_regularization_losses
Atrainable_variables
\metrics
]layer_metrics

^layers
 
 
 
 
 

0
 
 
 
 
 
 
 
 
 
 
4
	_total
	`count
a	variables
b	keras_api
D
	ctotal
	dcount
e
_fn_kwargs
f	variables
g	keras_api
 
 
 
 
 
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

_0
`1

a	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

c0
d1

f	variables
}{
VARIABLE_VALUEAdam/dense_2/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_2/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_3/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_3/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE Adam/lstm_2/lstm_cell_2/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*Adam/lstm_2/lstm_cell_2/recurrent_kernel/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/lstm_2/lstm_cell_2/bias/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE Adam/lstm_3/lstm_cell_3/kernel/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*Adam/lstm_3/lstm_cell_3/recurrent_kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/lstm_3/lstm_cell_3/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_2/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_2/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_3/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_3/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE Adam/lstm_2/lstm_cell_2/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*Adam/lstm_2/lstm_cell_2/recurrent_kernel/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/lstm_2/lstm_cell_2/bias/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE Adam/lstm_3/lstm_cell_3/kernel/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*Adam/lstm_3/lstm_cell_3/recurrent_kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/lstm_3/lstm_cell_3/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_lstm_2_inputPlaceholder*+
_output_shapes
:?????????*
dtype0* 
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_lstm_2_inputlstm_2/lstm_cell_2/kernellstm_2/lstm_cell_2/bias#lstm_2/lstm_cell_2/recurrent_kernellstm_3/lstm_cell_3/kernellstm_3/lstm_cell_3/bias#lstm_3/lstm_cell_3/recurrent_kerneldense_2/kerneldense_2/biasdense_3/kerneldense_3/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *,
f'R%
#__inference_signature_wrapper_56227
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename"dense_2/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOp"dense_3/kernel/Read/ReadVariableOp dense_3/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp-lstm_2/lstm_cell_2/kernel/Read/ReadVariableOp7lstm_2/lstm_cell_2/recurrent_kernel/Read/ReadVariableOp+lstm_2/lstm_cell_2/bias/Read/ReadVariableOp-lstm_3/lstm_cell_3/kernel/Read/ReadVariableOp7lstm_3/lstm_cell_3/recurrent_kernel/Read/ReadVariableOp+lstm_3/lstm_cell_3/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp)Adam/dense_2/kernel/m/Read/ReadVariableOp'Adam/dense_2/bias/m/Read/ReadVariableOp)Adam/dense_3/kernel/m/Read/ReadVariableOp'Adam/dense_3/bias/m/Read/ReadVariableOp4Adam/lstm_2/lstm_cell_2/kernel/m/Read/ReadVariableOp>Adam/lstm_2/lstm_cell_2/recurrent_kernel/m/Read/ReadVariableOp2Adam/lstm_2/lstm_cell_2/bias/m/Read/ReadVariableOp4Adam/lstm_3/lstm_cell_3/kernel/m/Read/ReadVariableOp>Adam/lstm_3/lstm_cell_3/recurrent_kernel/m/Read/ReadVariableOp2Adam/lstm_3/lstm_cell_3/bias/m/Read/ReadVariableOp)Adam/dense_2/kernel/v/Read/ReadVariableOp'Adam/dense_2/bias/v/Read/ReadVariableOp)Adam/dense_3/kernel/v/Read/ReadVariableOp'Adam/dense_3/bias/v/Read/ReadVariableOp4Adam/lstm_2/lstm_cell_2/kernel/v/Read/ReadVariableOp>Adam/lstm_2/lstm_cell_2/recurrent_kernel/v/Read/ReadVariableOp2Adam/lstm_2/lstm_cell_2/bias/v/Read/ReadVariableOp4Adam/lstm_3/lstm_cell_3/kernel/v/Read/ReadVariableOp>Adam/lstm_3/lstm_cell_3/recurrent_kernel/v/Read/ReadVariableOp2Adam/lstm_3/lstm_cell_3/bias/v/Read/ReadVariableOpConst*4
Tin-
+2)	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *'
f"R 
__inference__traced_save_60061
?	
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_2/kerneldense_2/biasdense_3/kerneldense_3/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratelstm_2/lstm_cell_2/kernel#lstm_2/lstm_cell_2/recurrent_kernellstm_2/lstm_cell_2/biaslstm_3/lstm_cell_3/kernel#lstm_3/lstm_cell_3/recurrent_kernellstm_3/lstm_cell_3/biastotalcounttotal_1count_1Adam/dense_2/kernel/mAdam/dense_2/bias/mAdam/dense_3/kernel/mAdam/dense_3/bias/m Adam/lstm_2/lstm_cell_2/kernel/m*Adam/lstm_2/lstm_cell_2/recurrent_kernel/mAdam/lstm_2/lstm_cell_2/bias/m Adam/lstm_3/lstm_cell_3/kernel/m*Adam/lstm_3/lstm_cell_3/recurrent_kernel/mAdam/lstm_3/lstm_cell_3/bias/mAdam/dense_2/kernel/vAdam/dense_2/bias/vAdam/dense_3/kernel/vAdam/dense_3/bias/v Adam/lstm_2/lstm_cell_2/kernel/v*Adam/lstm_2/lstm_cell_2/recurrent_kernel/vAdam/lstm_2/lstm_cell_2/bias/v Adam/lstm_3/lstm_cell_3/kernel/v*Adam/lstm_3/lstm_cell_3/recurrent_kernel/vAdam/lstm_3/lstm_cell_3/bias/v*3
Tin,
*2(*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? **
f%R#
!__inference__traced_restore_60188??4
?
?
while_cond_54989
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_54989___redundant_placeholder03
/while_while_cond_54989___redundant_placeholder13
/while_while_cond_54989___redundant_placeholder23
/while_while_cond_54989___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?J
?
F__inference_lstm_cell_3_layer_call_and_return_conditional_losses_54530

inputs

states
states_11
split_readvariableop_resource:
??.
split_1_readvariableop_resource:	?+
readvariableop_resource:
??
identity

identity_1

identity_2??ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?ReadVariableOp_3?split/ReadVariableOp?split_1/ReadVariableOpQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :t
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
??:
??:
??:
??*
	num_split[
MatMulMatMulinputssplit:output:0*
T0*(
_output_shapes
:??????????]
MatMul_1MatMulinputssplit:output:1*
T0*(
_output_shapes
:??????????]
MatMul_2MatMulinputssplit:output:2*
T0*(
_output_shapes
:??????????]
MatMul_3MatMulinputssplit:output:3*
T0*(
_output_shapes
:??????????S
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : s
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes	
:?*
dtype0?
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_spliti
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*(
_output_shapes
:??????????m
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*(
_output_shapes
:??????????m
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*(
_output_shapes
:??????????m
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*(
_output_shapes
:??????????h
ReadVariableOpReadVariableOpreadvariableop_resource* 
_output_shapes
:
??*
dtype0d
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_maske
MatMul_4MatMulstatesstrided_slice:output:0*
T0*(
_output_shapes
:??????????e
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*(
_output_shapes
:??????????J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>L
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?V
MulMuladd:z:0Const:output:0*
T0*(
_output_shapes
:??????????\
Add_1AddV2Mul:z:0Const_1:output:0*
T0*(
_output_shapes
:??????????\
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:??????????T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*(
_output_shapes
:??????????j
ReadVariableOp_1ReadVariableOpreadvariableop_resource* 
_output_shapes
:
??*
dtype0f
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   h
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_maskg
MatMul_5MatMulstatesstrided_slice_1:output:0*
T0*(
_output_shapes
:??????????i
add_2AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*(
_output_shapes
:??????????L
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>L
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?\
Mul_1Mul	add_2:z:0Const_2:output:0*
T0*(
_output_shapes
:??????????^
Add_3AddV2	Mul_1:z:0Const_3:output:0*
T0*(
_output_shapes
:??????????^
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:??????????V
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*(
_output_shapes
:??????????^
mul_2Mulclip_by_value_1:z:0states_1*
T0*(
_output_shapes
:??????????j
ReadVariableOp_2ReadVariableOpreadvariableop_resource* 
_output_shapes
:
??*
dtype0f
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  h
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_maskg
MatMul_6MatMulstatesstrided_slice_2:output:0*
T0*(
_output_shapes
:??????????i
add_4AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*(
_output_shapes
:??????????J
TanhTanh	add_4:z:0*
T0*(
_output_shapes
:??????????\
mul_3Mulclip_by_value:z:0Tanh:y:0*
T0*(
_output_shapes
:??????????W
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*(
_output_shapes
:??????????j
ReadVariableOp_3ReadVariableOpreadvariableop_resource* 
_output_shapes
:
??*
dtype0f
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  h
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_maskg
MatMul_7MatMulstatesstrided_slice_3:output:0*
T0*(
_output_shapes
:??????????i
add_6AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*(
_output_shapes
:??????????L
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>L
Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?\
Mul_4Mul	add_6:z:0Const_4:output:0*
T0*(
_output_shapes
:??????????^
Add_7AddV2	Mul_4:z:0Const_5:output:0*
T0*(
_output_shapes
:??????????^
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:??????????V
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*(
_output_shapes
:??????????L
Tanh_1Tanh	add_5:z:0*
T0*(
_output_shapes
:??????????`
mul_5Mulclip_by_value_2:z:0
Tanh_1:y:0*
T0*(
_output_shapes
:??????????Y
IdentityIdentity	mul_5:z:0^NoOp*
T0*(
_output_shapes
:??????????[

Identity_1Identity	mul_5:z:0^NoOp*
T0*(
_output_shapes
:??????????[

Identity_2Identity	add_5:z:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:??????????:??????????:??????????: : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:PL
(
_output_shapes
:??????????
 
_user_specified_namestates:PL
(
_output_shapes
:??????????
 
_user_specified_namestates
?"
?
while_body_54791
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0-
while_lstm_cell_3_54815_0:
??(
while_lstm_cell_3_54817_0:	?-
while_lstm_cell_3_54819_0:
??
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor+
while_lstm_cell_3_54815:
??&
while_lstm_cell_3_54817:	?+
while_lstm_cell_3_54819:
????)while/lstm_cell_3/StatefulPartitionedCall?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype0?
)while/lstm_cell_3/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_3_54815_0while_lstm_cell_3_54817_0while_lstm_cell_3_54819_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:??????????:??????????:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_lstm_cell_3_layer_call_and_return_conditional_losses_54732?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder2while/lstm_cell_3/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:???M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_4Identity2while/lstm_cell_3/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:???????????
while/Identity_5Identity2while/lstm_cell_3/StatefulPartitionedCall:output:2^while/NoOp*
T0*(
_output_shapes
:??????????x

while/NoOpNoOp*^while/lstm_cell_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"4
while_lstm_cell_3_54815while_lstm_cell_3_54815_0"4
while_lstm_cell_3_54817while_lstm_cell_3_54817_0"4
while_lstm_cell_3_54819while_lstm_cell_3_54819_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2V
)while/lstm_cell_3/StatefulPartitionedCall)while/lstm_cell_3/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?K
?
F__inference_lstm_cell_3_layer_call_and_return_conditional_losses_59887

inputs
states_0
states_11
split_readvariableop_resource:
??.
split_1_readvariableop_resource:	?+
readvariableop_resource:
??
identity

identity_1

identity_2??ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?ReadVariableOp_3?split/ReadVariableOp?split_1/ReadVariableOpQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :t
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
??:
??:
??:
??*
	num_split[
MatMulMatMulinputssplit:output:0*
T0*(
_output_shapes
:??????????]
MatMul_1MatMulinputssplit:output:1*
T0*(
_output_shapes
:??????????]
MatMul_2MatMulinputssplit:output:2*
T0*(
_output_shapes
:??????????]
MatMul_3MatMulinputssplit:output:3*
T0*(
_output_shapes
:??????????S
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : s
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes	
:?*
dtype0?
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_spliti
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*(
_output_shapes
:??????????m
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*(
_output_shapes
:??????????m
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*(
_output_shapes
:??????????m
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*(
_output_shapes
:??????????h
ReadVariableOpReadVariableOpreadvariableop_resource* 
_output_shapes
:
??*
dtype0d
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_maskg
MatMul_4MatMulstates_0strided_slice:output:0*
T0*(
_output_shapes
:??????????e
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*(
_output_shapes
:??????????J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>L
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?V
MulMuladd:z:0Const:output:0*
T0*(
_output_shapes
:??????????\
Add_1AddV2Mul:z:0Const_1:output:0*
T0*(
_output_shapes
:??????????\
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:??????????T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*(
_output_shapes
:??????????j
ReadVariableOp_1ReadVariableOpreadvariableop_resource* 
_output_shapes
:
??*
dtype0f
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   h
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_maski
MatMul_5MatMulstates_0strided_slice_1:output:0*
T0*(
_output_shapes
:??????????i
add_2AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*(
_output_shapes
:??????????L
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>L
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?\
Mul_1Mul	add_2:z:0Const_2:output:0*
T0*(
_output_shapes
:??????????^
Add_3AddV2	Mul_1:z:0Const_3:output:0*
T0*(
_output_shapes
:??????????^
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:??????????V
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*(
_output_shapes
:??????????^
mul_2Mulclip_by_value_1:z:0states_1*
T0*(
_output_shapes
:??????????j
ReadVariableOp_2ReadVariableOpreadvariableop_resource* 
_output_shapes
:
??*
dtype0f
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  h
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_maski
MatMul_6MatMulstates_0strided_slice_2:output:0*
T0*(
_output_shapes
:??????????i
add_4AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*(
_output_shapes
:??????????J
TanhTanh	add_4:z:0*
T0*(
_output_shapes
:??????????\
mul_3Mulclip_by_value:z:0Tanh:y:0*
T0*(
_output_shapes
:??????????W
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*(
_output_shapes
:??????????j
ReadVariableOp_3ReadVariableOpreadvariableop_resource* 
_output_shapes
:
??*
dtype0f
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  h
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_maski
MatMul_7MatMulstates_0strided_slice_3:output:0*
T0*(
_output_shapes
:??????????i
add_6AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*(
_output_shapes
:??????????L
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>L
Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?\
Mul_4Mul	add_6:z:0Const_4:output:0*
T0*(
_output_shapes
:??????????^
Add_7AddV2	Mul_4:z:0Const_5:output:0*
T0*(
_output_shapes
:??????????^
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:??????????V
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*(
_output_shapes
:??????????L
Tanh_1Tanh	add_5:z:0*
T0*(
_output_shapes
:??????????`
mul_5Mulclip_by_value_2:z:0
Tanh_1:y:0*
T0*(
_output_shapes
:??????????Y
IdentityIdentity	mul_5:z:0^NoOp*
T0*(
_output_shapes
:??????????[

Identity_1Identity	mul_5:z:0^NoOp*
T0*(
_output_shapes
:??????????[

Identity_2Identity	add_5:z:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:??????????:??????????:??????????: : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/0:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/1
?	
?
lstm_2_while_cond_56342*
&lstm_2_while_lstm_2_while_loop_counter0
,lstm_2_while_lstm_2_while_maximum_iterations
lstm_2_while_placeholder
lstm_2_while_placeholder_1
lstm_2_while_placeholder_2
lstm_2_while_placeholder_3,
(lstm_2_while_less_lstm_2_strided_slice_1A
=lstm_2_while_lstm_2_while_cond_56342___redundant_placeholder0A
=lstm_2_while_lstm_2_while_cond_56342___redundant_placeholder1A
=lstm_2_while_lstm_2_while_cond_56342___redundant_placeholder2A
=lstm_2_while_lstm_2_while_cond_56342___redundant_placeholder3
lstm_2_while_identity
~
lstm_2/while/LessLesslstm_2_while_placeholder(lstm_2_while_less_lstm_2_strided_slice_1*
T0*
_output_shapes
: Y
lstm_2/while/IdentityIdentitylstm_2/while/Less:z:0*
T0
*
_output_shapes
: "7
lstm_2_while_identitylstm_2/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?
?
while_cond_54790
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_54790___redundant_placeholder03
/while_while_cond_54790___redundant_placeholder13
/while_while_cond_54790___redundant_placeholder23
/while_while_cond_54790___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?
?
&__inference_lstm_3_layer_call_fn_59446

inputs
unknown:
??
	unknown_0:	?
	unknown_1:
??
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_lstm_3_layer_call_and_return_conditional_losses_55393p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:??????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
G__inference_sequential_1_layer_call_and_return_conditional_losses_56166
lstm_2_input
lstm_2_56141:	?
lstm_2_56143:	? 
lstm_2_56145:
?? 
lstm_3_56148:
??
lstm_3_56150:	? 
lstm_3_56152:
??!
dense_2_56155:
??
dense_2_56157:	? 
dense_3_56160:	?
dense_3_56162:
identity??dense_2/StatefulPartitionedCall?dense_3/StatefulPartitionedCall?lstm_2/StatefulPartitionedCall?lstm_3/StatefulPartitionedCall?
lstm_2/StatefulPartitionedCallStatefulPartitionedCalllstm_2_inputlstm_2_56141lstm_2_56143lstm_2_56145*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_lstm_2_layer_call_and_return_conditional_losses_55130?
lstm_3/StatefulPartitionedCallStatefulPartitionedCall'lstm_2/StatefulPartitionedCall:output:0lstm_3_56148lstm_3_56150lstm_3_56152*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_lstm_3_layer_call_and_return_conditional_losses_55393?
dense_2/StatefulPartitionedCallStatefulPartitionedCall'lstm_3/StatefulPartitionedCall:output:0dense_2_56155dense_2_56157*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_55412?
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_56160dense_3_56162*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_55429w
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall^lstm_2/StatefulPartitionedCall^lstm_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????: : : : : : : : : : 2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2@
lstm_2/StatefulPartitionedCalllstm_2/StatefulPartitionedCall2@
lstm_3/StatefulPartitionedCalllstm_3/StatefulPartitionedCall:Y U
+
_output_shapes
:?????????
&
_user_specified_namelstm_2_input
?

?
,__inference_sequential_1_layer_call_fn_57296

inputs
unknown:	?
	unknown_0:	?
	unknown_1:
??
	unknown_2:
??
	unknown_3:	?
	unknown_4:
??
	unknown_5:
??
	unknown_6:	?
	unknown_7:	?
	unknown_8:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_55436o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
,__inference_sequential_1_layer_call_fn_55459
lstm_2_input
unknown:	?
	unknown_0:	?
	unknown_1:
??
	unknown_2:
??
	unknown_3:	?
	unknown_4:
??
	unknown_5:
??
	unknown_6:	?
	unknown_7:	?
	unknown_8:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalllstm_2_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_55436o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
+
_output_shapes
:?????????
&
_user_specified_namelstm_2_input
?7
?
A__inference_lstm_2_layer_call_and_return_conditional_losses_54150

inputs$
lstm_cell_2_54069:	? 
lstm_cell_2_54071:	?%
lstm_cell_2_54073:
??
identity??#lstm_cell_2/StatefulPartitionedCall?while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????S
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask?
#lstm_cell_2/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_2_54069lstm_cell_2_54071lstm_cell_2_54073*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:??????????:??????????:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_lstm_cell_2_layer_call_and_return_conditional_losses_54068n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_2_54069lstm_cell_2_54071lstm_cell_2_54073*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_54082*
condR
while_cond_54081*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:???????????????????*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:???????????????????l
IdentityIdentitytranspose_1:y:0^NoOp*
T0*5
_output_shapes#
!:???????????????????t
NoOpNoOp$^lstm_cell_2/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????: : : 2J
#lstm_cell_2/StatefulPartitionedCall#lstm_cell_2/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?J
?
F__inference_lstm_cell_3_layer_call_and_return_conditional_losses_54732

inputs

states
states_11
split_readvariableop_resource:
??.
split_1_readvariableop_resource:	?+
readvariableop_resource:
??
identity

identity_1

identity_2??ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?ReadVariableOp_3?split/ReadVariableOp?split_1/ReadVariableOpQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :t
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
??:
??:
??:
??*
	num_split[
MatMulMatMulinputssplit:output:0*
T0*(
_output_shapes
:??????????]
MatMul_1MatMulinputssplit:output:1*
T0*(
_output_shapes
:??????????]
MatMul_2MatMulinputssplit:output:2*
T0*(
_output_shapes
:??????????]
MatMul_3MatMulinputssplit:output:3*
T0*(
_output_shapes
:??????????S
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : s
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes	
:?*
dtype0?
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_spliti
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*(
_output_shapes
:??????????m
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*(
_output_shapes
:??????????m
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*(
_output_shapes
:??????????m
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*(
_output_shapes
:??????????h
ReadVariableOpReadVariableOpreadvariableop_resource* 
_output_shapes
:
??*
dtype0d
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_maske
MatMul_4MatMulstatesstrided_slice:output:0*
T0*(
_output_shapes
:??????????e
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*(
_output_shapes
:??????????J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>L
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?V
MulMuladd:z:0Const:output:0*
T0*(
_output_shapes
:??????????\
Add_1AddV2Mul:z:0Const_1:output:0*
T0*(
_output_shapes
:??????????\
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:??????????T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*(
_output_shapes
:??????????j
ReadVariableOp_1ReadVariableOpreadvariableop_resource* 
_output_shapes
:
??*
dtype0f
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   h
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_maskg
MatMul_5MatMulstatesstrided_slice_1:output:0*
T0*(
_output_shapes
:??????????i
add_2AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*(
_output_shapes
:??????????L
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>L
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?\
Mul_1Mul	add_2:z:0Const_2:output:0*
T0*(
_output_shapes
:??????????^
Add_3AddV2	Mul_1:z:0Const_3:output:0*
T0*(
_output_shapes
:??????????^
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:??????????V
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*(
_output_shapes
:??????????^
mul_2Mulclip_by_value_1:z:0states_1*
T0*(
_output_shapes
:??????????j
ReadVariableOp_2ReadVariableOpreadvariableop_resource* 
_output_shapes
:
??*
dtype0f
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  h
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_maskg
MatMul_6MatMulstatesstrided_slice_2:output:0*
T0*(
_output_shapes
:??????????i
add_4AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*(
_output_shapes
:??????????J
TanhTanh	add_4:z:0*
T0*(
_output_shapes
:??????????\
mul_3Mulclip_by_value:z:0Tanh:y:0*
T0*(
_output_shapes
:??????????W
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*(
_output_shapes
:??????????j
ReadVariableOp_3ReadVariableOpreadvariableop_resource* 
_output_shapes
:
??*
dtype0f
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  h
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_maskg
MatMul_7MatMulstatesstrided_slice_3:output:0*
T0*(
_output_shapes
:??????????i
add_6AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*(
_output_shapes
:??????????L
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>L
Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?\
Mul_4Mul	add_6:z:0Const_4:output:0*
T0*(
_output_shapes
:??????????^
Add_7AddV2	Mul_4:z:0Const_5:output:0*
T0*(
_output_shapes
:??????????^
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:??????????V
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*(
_output_shapes
:??????????L
Tanh_1Tanh	add_5:z:0*
T0*(
_output_shapes
:??????????`
mul_5Mulclip_by_value_2:z:0
Tanh_1:y:0*
T0*(
_output_shapes
:??????????Y
IdentityIdentity	mul_5:z:0^NoOp*
T0*(
_output_shapes
:??????????[

Identity_1Identity	mul_5:z:0^NoOp*
T0*(
_output_shapes
:??????????[

Identity_2Identity	add_5:z:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:??????????:??????????:??????????: : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:PL
(
_output_shapes
:??????????
 
_user_specified_namestates:PL
(
_output_shapes
:??????????
 
_user_specified_namestates
?

?
B__inference_dense_3_layer_call_and_return_conditional_losses_59488

inputs1
matmul_readvariableop_resource:	?-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?K
?
F__inference_lstm_cell_2_layer_call_and_return_conditional_losses_59586

inputs
states_0
states_10
split_readvariableop_resource:	?.
split_1_readvariableop_resource:	?+
readvariableop_resource:
??
identity

identity_1

identity_2??ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?ReadVariableOp_3?split/ReadVariableOp?split_1/ReadVariableOpQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :s
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes
:	?*
dtype0?
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	?:	?:	?:	?*
	num_split[
MatMulMatMulinputssplit:output:0*
T0*(
_output_shapes
:??????????]
MatMul_1MatMulinputssplit:output:1*
T0*(
_output_shapes
:??????????]
MatMul_2MatMulinputssplit:output:2*
T0*(
_output_shapes
:??????????]
MatMul_3MatMulinputssplit:output:3*
T0*(
_output_shapes
:??????????S
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : s
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes	
:?*
dtype0?
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_spliti
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*(
_output_shapes
:??????????m
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*(
_output_shapes
:??????????m
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*(
_output_shapes
:??????????m
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*(
_output_shapes
:??????????h
ReadVariableOpReadVariableOpreadvariableop_resource* 
_output_shapes
:
??*
dtype0d
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_maskg
MatMul_4MatMulstates_0strided_slice:output:0*
T0*(
_output_shapes
:??????????e
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*(
_output_shapes
:??????????J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>L
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?V
MulMuladd:z:0Const:output:0*
T0*(
_output_shapes
:??????????\
Add_1AddV2Mul:z:0Const_1:output:0*
T0*(
_output_shapes
:??????????\
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:??????????T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*(
_output_shapes
:??????????j
ReadVariableOp_1ReadVariableOpreadvariableop_resource* 
_output_shapes
:
??*
dtype0f
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_maski
MatMul_5MatMulstates_0strided_slice_1:output:0*
T0*(
_output_shapes
:??????????i
add_2AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*(
_output_shapes
:??????????L
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>L
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?\
Mul_1Mul	add_2:z:0Const_2:output:0*
T0*(
_output_shapes
:??????????^
Add_3AddV2	Mul_1:z:0Const_3:output:0*
T0*(
_output_shapes
:??????????^
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:??????????V
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*(
_output_shapes
:??????????^
mul_2Mulclip_by_value_1:z:0states_1*
T0*(
_output_shapes
:??????????j
ReadVariableOp_2ReadVariableOpreadvariableop_resource* 
_output_shapes
:
??*
dtype0f
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_maski
MatMul_6MatMulstates_0strided_slice_2:output:0*
T0*(
_output_shapes
:??????????i
add_4AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*(
_output_shapes
:??????????J
TanhTanh	add_4:z:0*
T0*(
_output_shapes
:??????????\
mul_3Mulclip_by_value:z:0Tanh:y:0*
T0*(
_output_shapes
:??????????W
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*(
_output_shapes
:??????????j
ReadVariableOp_3ReadVariableOpreadvariableop_resource* 
_output_shapes
:
??*
dtype0f
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_maski
MatMul_7MatMulstates_0strided_slice_3:output:0*
T0*(
_output_shapes
:??????????i
add_6AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*(
_output_shapes
:??????????L
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>L
Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?\
Mul_4Mul	add_6:z:0Const_4:output:0*
T0*(
_output_shapes
:??????????^
Add_7AddV2	Mul_4:z:0Const_5:output:0*
T0*(
_output_shapes
:??????????^
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:??????????V
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*(
_output_shapes
:??????????L
Tanh_1Tanh	add_5:z:0*
T0*(
_output_shapes
:??????????`
mul_5Mulclip_by_value_2:z:0
Tanh_1:y:0*
T0*(
_output_shapes
:??????????Y
IdentityIdentity	mul_5:z:0^NoOp*
T0*(
_output_shapes
:??????????[

Identity_1Identity	mul_5:z:0^NoOp*
T0*(
_output_shapes
:??????????[

Identity_2Identity	add_5:z:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:?????????:??????????:??????????: : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/0:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/1
?
?
+__inference_lstm_cell_2_layer_call_fn_59692

inputs
states_0
states_1
unknown:	?
	unknown_0:	?
	unknown_1:
??
identity

identity_1

identity_2??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:??????????:??????????:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_lstm_cell_2_layer_call_and_return_conditional_losses_54068p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????r

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:??????????r

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:?????????:??????????:??????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/0:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/1
?{
?	
while_body_57437
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_lstm_cell_2_split_readvariableop_resource_0:	?B
3while_lstm_cell_2_split_1_readvariableop_resource_0:	??
+while_lstm_cell_2_readvariableop_resource_0:
??
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_lstm_cell_2_split_readvariableop_resource:	?@
1while_lstm_cell_2_split_1_readvariableop_resource:	?=
)while_lstm_cell_2_readvariableop_resource:
???? while/lstm_cell_2/ReadVariableOp?"while/lstm_cell_2/ReadVariableOp_1?"while/lstm_cell_2/ReadVariableOp_2?"while/lstm_cell_2/ReadVariableOp_3?&while/lstm_cell_2/split/ReadVariableOp?(while/lstm_cell_2/split_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0c
!while/lstm_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
&while/lstm_cell_2/split/ReadVariableOpReadVariableOp1while_lstm_cell_2_split_readvariableop_resource_0*
_output_shapes
:	?*
dtype0?
while/lstm_cell_2/splitSplit*while/lstm_cell_2/split/split_dim:output:0.while/lstm_cell_2/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	?:	?:	?:	?*
	num_split?
while/lstm_cell_2/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_2/split:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_2/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_2/split:output:1*
T0*(
_output_shapes
:???????????
while/lstm_cell_2/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_2/split:output:2*
T0*(
_output_shapes
:???????????
while/lstm_cell_2/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_2/split:output:3*
T0*(
_output_shapes
:??????????e
#while/lstm_cell_2/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
(while/lstm_cell_2/split_1/ReadVariableOpReadVariableOp3while_lstm_cell_2_split_1_readvariableop_resource_0*
_output_shapes	
:?*
dtype0?
while/lstm_cell_2/split_1Split,while/lstm_cell_2/split_1/split_dim:output:00while/lstm_cell_2/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split?
while/lstm_cell_2/BiasAddBiasAdd"while/lstm_cell_2/MatMul:product:0"while/lstm_cell_2/split_1:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_2/BiasAdd_1BiasAdd$while/lstm_cell_2/MatMul_1:product:0"while/lstm_cell_2/split_1:output:1*
T0*(
_output_shapes
:???????????
while/lstm_cell_2/BiasAdd_2BiasAdd$while/lstm_cell_2/MatMul_2:product:0"while/lstm_cell_2/split_1:output:2*
T0*(
_output_shapes
:???????????
while/lstm_cell_2/BiasAdd_3BiasAdd$while/lstm_cell_2/MatMul_3:product:0"while/lstm_cell_2/split_1:output:3*
T0*(
_output_shapes
:???????????
 while/lstm_cell_2/ReadVariableOpReadVariableOp+while_lstm_cell_2_readvariableop_resource_0* 
_output_shapes
:
??*
dtype0v
%while/lstm_cell_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        x
'while/lstm_cell_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       x
'while/lstm_cell_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
while/lstm_cell_2/strided_sliceStridedSlice(while/lstm_cell_2/ReadVariableOp:value:0.while/lstm_cell_2/strided_slice/stack:output:00while/lstm_cell_2/strided_slice/stack_1:output:00while/lstm_cell_2/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
while/lstm_cell_2/MatMul_4MatMulwhile_placeholder_2(while/lstm_cell_2/strided_slice:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_2/addAddV2"while/lstm_cell_2/BiasAdd:output:0$while/lstm_cell_2/MatMul_4:product:0*
T0*(
_output_shapes
:??????????\
while/lstm_cell_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>^
while/lstm_cell_2/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ??
while/lstm_cell_2/MulMulwhile/lstm_cell_2/add:z:0 while/lstm_cell_2/Const:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_2/Add_1AddV2while/lstm_cell_2/Mul:z:0"while/lstm_cell_2/Const_1:output:0*
T0*(
_output_shapes
:??????????n
)while/lstm_cell_2/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
'while/lstm_cell_2/clip_by_value/MinimumMinimumwhile/lstm_cell_2/Add_1:z:02while/lstm_cell_2/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:??????????f
!while/lstm_cell_2/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
while/lstm_cell_2/clip_by_valueMaximum+while/lstm_cell_2/clip_by_value/Minimum:z:0*while/lstm_cell_2/clip_by_value/y:output:0*
T0*(
_output_shapes
:???????????
"while/lstm_cell_2/ReadVariableOp_1ReadVariableOp+while_lstm_cell_2_readvariableop_resource_0* 
_output_shapes
:
??*
dtype0x
'while/lstm_cell_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       z
)while/lstm_cell_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       z
)while/lstm_cell_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
!while/lstm_cell_2/strided_slice_1StridedSlice*while/lstm_cell_2/ReadVariableOp_1:value:00while/lstm_cell_2/strided_slice_1/stack:output:02while/lstm_cell_2/strided_slice_1/stack_1:output:02while/lstm_cell_2/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
while/lstm_cell_2/MatMul_5MatMulwhile_placeholder_2*while/lstm_cell_2/strided_slice_1:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_2/add_2AddV2$while/lstm_cell_2/BiasAdd_1:output:0$while/lstm_cell_2/MatMul_5:product:0*
T0*(
_output_shapes
:??????????^
while/lstm_cell_2/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>^
while/lstm_cell_2/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ??
while/lstm_cell_2/Mul_1Mulwhile/lstm_cell_2/add_2:z:0"while/lstm_cell_2/Const_2:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_2/Add_3AddV2while/lstm_cell_2/Mul_1:z:0"while/lstm_cell_2/Const_3:output:0*
T0*(
_output_shapes
:??????????p
+while/lstm_cell_2/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
)while/lstm_cell_2/clip_by_value_1/MinimumMinimumwhile/lstm_cell_2/Add_3:z:04while/lstm_cell_2/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:??????????h
#while/lstm_cell_2/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
!while/lstm_cell_2/clip_by_value_1Maximum-while/lstm_cell_2/clip_by_value_1/Minimum:z:0,while/lstm_cell_2/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_2/mul_2Mul%while/lstm_cell_2/clip_by_value_1:z:0while_placeholder_3*
T0*(
_output_shapes
:???????????
"while/lstm_cell_2/ReadVariableOp_2ReadVariableOp+while_lstm_cell_2_readvariableop_resource_0* 
_output_shapes
:
??*
dtype0x
'while/lstm_cell_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       z
)while/lstm_cell_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       z
)while/lstm_cell_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
!while/lstm_cell_2/strided_slice_2StridedSlice*while/lstm_cell_2/ReadVariableOp_2:value:00while/lstm_cell_2/strided_slice_2/stack:output:02while/lstm_cell_2/strided_slice_2/stack_1:output:02while/lstm_cell_2/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
while/lstm_cell_2/MatMul_6MatMulwhile_placeholder_2*while/lstm_cell_2/strided_slice_2:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_2/add_4AddV2$while/lstm_cell_2/BiasAdd_2:output:0$while/lstm_cell_2/MatMul_6:product:0*
T0*(
_output_shapes
:??????????n
while/lstm_cell_2/TanhTanhwhile/lstm_cell_2/add_4:z:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_2/mul_3Mul#while/lstm_cell_2/clip_by_value:z:0while/lstm_cell_2/Tanh:y:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_2/add_5AddV2while/lstm_cell_2/mul_2:z:0while/lstm_cell_2/mul_3:z:0*
T0*(
_output_shapes
:???????????
"while/lstm_cell_2/ReadVariableOp_3ReadVariableOp+while_lstm_cell_2_readvariableop_resource_0* 
_output_shapes
:
??*
dtype0x
'while/lstm_cell_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       z
)while/lstm_cell_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        z
)while/lstm_cell_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
!while/lstm_cell_2/strided_slice_3StridedSlice*while/lstm_cell_2/ReadVariableOp_3:value:00while/lstm_cell_2/strided_slice_3/stack:output:02while/lstm_cell_2/strided_slice_3/stack_1:output:02while/lstm_cell_2/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
while/lstm_cell_2/MatMul_7MatMulwhile_placeholder_2*while/lstm_cell_2/strided_slice_3:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_2/add_6AddV2$while/lstm_cell_2/BiasAdd_3:output:0$while/lstm_cell_2/MatMul_7:product:0*
T0*(
_output_shapes
:??????????^
while/lstm_cell_2/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>^
while/lstm_cell_2/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ??
while/lstm_cell_2/Mul_4Mulwhile/lstm_cell_2/add_6:z:0"while/lstm_cell_2/Const_4:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_2/Add_7AddV2while/lstm_cell_2/Mul_4:z:0"while/lstm_cell_2/Const_5:output:0*
T0*(
_output_shapes
:??????????p
+while/lstm_cell_2/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
)while/lstm_cell_2/clip_by_value_2/MinimumMinimumwhile/lstm_cell_2/Add_7:z:04while/lstm_cell_2/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:??????????h
#while/lstm_cell_2/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
!while/lstm_cell_2/clip_by_value_2Maximum-while/lstm_cell_2/clip_by_value_2/Minimum:z:0,while/lstm_cell_2/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:??????????p
while/lstm_cell_2/Tanh_1Tanhwhile/lstm_cell_2/add_5:z:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_2/mul_5Mul%while/lstm_cell_2/clip_by_value_2:z:0while/lstm_cell_2/Tanh_1:y:0*
T0*(
_output_shapes
:???????????
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_2/mul_5:z:0*
_output_shapes
: *
element_dtype0:???M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: y
while/Identity_4Identitywhile/lstm_cell_2/mul_5:z:0^while/NoOp*
T0*(
_output_shapes
:??????????y
while/Identity_5Identitywhile/lstm_cell_2/add_5:z:0^while/NoOp*
T0*(
_output_shapes
:???????????

while/NoOpNoOp!^while/lstm_cell_2/ReadVariableOp#^while/lstm_cell_2/ReadVariableOp_1#^while/lstm_cell_2/ReadVariableOp_2#^while/lstm_cell_2/ReadVariableOp_3'^while/lstm_cell_2/split/ReadVariableOp)^while/lstm_cell_2/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"X
)while_lstm_cell_2_readvariableop_resource+while_lstm_cell_2_readvariableop_resource_0"h
1while_lstm_cell_2_split_1_readvariableop_resource3while_lstm_cell_2_split_1_readvariableop_resource_0"d
/while_lstm_cell_2_split_readvariableop_resource1while_lstm_cell_2_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2D
 while/lstm_cell_2/ReadVariableOp while/lstm_cell_2/ReadVariableOp2H
"while/lstm_cell_2/ReadVariableOp_1"while/lstm_cell_2/ReadVariableOp_12H
"while/lstm_cell_2/ReadVariableOp_2"while/lstm_cell_2/ReadVariableOp_22H
"while/lstm_cell_2/ReadVariableOp_3"while/lstm_cell_2/ReadVariableOp_32P
&while/lstm_cell_2/split/ReadVariableOp&while/lstm_cell_2/split/ReadVariableOp2T
(while/lstm_cell_2/split_1/ReadVariableOp(while/lstm_cell_2/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
??
?
A__inference_lstm_2_layer_call_and_return_conditional_losses_58089

inputs<
)lstm_cell_2_split_readvariableop_resource:	?:
+lstm_cell_2_split_1_readvariableop_resource:	?7
#lstm_cell_2_readvariableop_resource:
??
identity??lstm_cell_2/ReadVariableOp?lstm_cell_2/ReadVariableOp_1?lstm_cell_2/ReadVariableOp_2?lstm_cell_2/ReadVariableOp_3? lstm_cell_2/split/ReadVariableOp?"lstm_cell_2/split_1/ReadVariableOp?while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????S
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:?????????D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask]
lstm_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
 lstm_cell_2/split/ReadVariableOpReadVariableOp)lstm_cell_2_split_readvariableop_resource*
_output_shapes
:	?*
dtype0?
lstm_cell_2/splitSplit$lstm_cell_2/split/split_dim:output:0(lstm_cell_2/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	?:	?:	?:	?*
	num_split?
lstm_cell_2/MatMulMatMulstrided_slice_2:output:0lstm_cell_2/split:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_2/MatMul_1MatMulstrided_slice_2:output:0lstm_cell_2/split:output:1*
T0*(
_output_shapes
:???????????
lstm_cell_2/MatMul_2MatMulstrided_slice_2:output:0lstm_cell_2/split:output:2*
T0*(
_output_shapes
:???????????
lstm_cell_2/MatMul_3MatMulstrided_slice_2:output:0lstm_cell_2/split:output:3*
T0*(
_output_shapes
:??????????_
lstm_cell_2/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
"lstm_cell_2/split_1/ReadVariableOpReadVariableOp+lstm_cell_2_split_1_readvariableop_resource*
_output_shapes	
:?*
dtype0?
lstm_cell_2/split_1Split&lstm_cell_2/split_1/split_dim:output:0*lstm_cell_2/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split?
lstm_cell_2/BiasAddBiasAddlstm_cell_2/MatMul:product:0lstm_cell_2/split_1:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_2/BiasAdd_1BiasAddlstm_cell_2/MatMul_1:product:0lstm_cell_2/split_1:output:1*
T0*(
_output_shapes
:???????????
lstm_cell_2/BiasAdd_2BiasAddlstm_cell_2/MatMul_2:product:0lstm_cell_2/split_1:output:2*
T0*(
_output_shapes
:???????????
lstm_cell_2/BiasAdd_3BiasAddlstm_cell_2/MatMul_3:product:0lstm_cell_2/split_1:output:3*
T0*(
_output_shapes
:???????????
lstm_cell_2/ReadVariableOpReadVariableOp#lstm_cell_2_readvariableop_resource* 
_output_shapes
:
??*
dtype0p
lstm_cell_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        r
!lstm_cell_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       r
!lstm_cell_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
lstm_cell_2/strided_sliceStridedSlice"lstm_cell_2/ReadVariableOp:value:0(lstm_cell_2/strided_slice/stack:output:0*lstm_cell_2/strided_slice/stack_1:output:0*lstm_cell_2/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
lstm_cell_2/MatMul_4MatMulzeros:output:0"lstm_cell_2/strided_slice:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_2/addAddV2lstm_cell_2/BiasAdd:output:0lstm_cell_2/MatMul_4:product:0*
T0*(
_output_shapes
:??????????V
lstm_cell_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>X
lstm_cell_2/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?z
lstm_cell_2/MulMullstm_cell_2/add:z:0lstm_cell_2/Const:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_2/Add_1AddV2lstm_cell_2/Mul:z:0lstm_cell_2/Const_1:output:0*
T0*(
_output_shapes
:??????????h
#lstm_cell_2/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
!lstm_cell_2/clip_by_value/MinimumMinimumlstm_cell_2/Add_1:z:0,lstm_cell_2/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:??????????`
lstm_cell_2/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
lstm_cell_2/clip_by_valueMaximum%lstm_cell_2/clip_by_value/Minimum:z:0$lstm_cell_2/clip_by_value/y:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_2/ReadVariableOp_1ReadVariableOp#lstm_cell_2_readvariableop_resource* 
_output_shapes
:
??*
dtype0r
!lstm_cell_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       t
#lstm_cell_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       t
#lstm_cell_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
lstm_cell_2/strided_slice_1StridedSlice$lstm_cell_2/ReadVariableOp_1:value:0*lstm_cell_2/strided_slice_1/stack:output:0,lstm_cell_2/strided_slice_1/stack_1:output:0,lstm_cell_2/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
lstm_cell_2/MatMul_5MatMulzeros:output:0$lstm_cell_2/strided_slice_1:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_2/add_2AddV2lstm_cell_2/BiasAdd_1:output:0lstm_cell_2/MatMul_5:product:0*
T0*(
_output_shapes
:??????????X
lstm_cell_2/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>X
lstm_cell_2/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ??
lstm_cell_2/Mul_1Mullstm_cell_2/add_2:z:0lstm_cell_2/Const_2:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_2/Add_3AddV2lstm_cell_2/Mul_1:z:0lstm_cell_2/Const_3:output:0*
T0*(
_output_shapes
:??????????j
%lstm_cell_2/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
#lstm_cell_2/clip_by_value_1/MinimumMinimumlstm_cell_2/Add_3:z:0.lstm_cell_2/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:??????????b
lstm_cell_2/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
lstm_cell_2/clip_by_value_1Maximum'lstm_cell_2/clip_by_value_1/Minimum:z:0&lstm_cell_2/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:??????????~
lstm_cell_2/mul_2Mullstm_cell_2/clip_by_value_1:z:0zeros_1:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_2/ReadVariableOp_2ReadVariableOp#lstm_cell_2_readvariableop_resource* 
_output_shapes
:
??*
dtype0r
!lstm_cell_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       t
#lstm_cell_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       t
#lstm_cell_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
lstm_cell_2/strided_slice_2StridedSlice$lstm_cell_2/ReadVariableOp_2:value:0*lstm_cell_2/strided_slice_2/stack:output:0,lstm_cell_2/strided_slice_2/stack_1:output:0,lstm_cell_2/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
lstm_cell_2/MatMul_6MatMulzeros:output:0$lstm_cell_2/strided_slice_2:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_2/add_4AddV2lstm_cell_2/BiasAdd_2:output:0lstm_cell_2/MatMul_6:product:0*
T0*(
_output_shapes
:??????????b
lstm_cell_2/TanhTanhlstm_cell_2/add_4:z:0*
T0*(
_output_shapes
:???????????
lstm_cell_2/mul_3Mullstm_cell_2/clip_by_value:z:0lstm_cell_2/Tanh:y:0*
T0*(
_output_shapes
:??????????{
lstm_cell_2/add_5AddV2lstm_cell_2/mul_2:z:0lstm_cell_2/mul_3:z:0*
T0*(
_output_shapes
:???????????
lstm_cell_2/ReadVariableOp_3ReadVariableOp#lstm_cell_2_readvariableop_resource* 
_output_shapes
:
??*
dtype0r
!lstm_cell_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       t
#lstm_cell_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        t
#lstm_cell_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
lstm_cell_2/strided_slice_3StridedSlice$lstm_cell_2/ReadVariableOp_3:value:0*lstm_cell_2/strided_slice_3/stack:output:0,lstm_cell_2/strided_slice_3/stack_1:output:0,lstm_cell_2/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
lstm_cell_2/MatMul_7MatMulzeros:output:0$lstm_cell_2/strided_slice_3:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_2/add_6AddV2lstm_cell_2/BiasAdd_3:output:0lstm_cell_2/MatMul_7:product:0*
T0*(
_output_shapes
:??????????X
lstm_cell_2/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>X
lstm_cell_2/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ??
lstm_cell_2/Mul_4Mullstm_cell_2/add_6:z:0lstm_cell_2/Const_4:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_2/Add_7AddV2lstm_cell_2/Mul_4:z:0lstm_cell_2/Const_5:output:0*
T0*(
_output_shapes
:??????????j
%lstm_cell_2/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
#lstm_cell_2/clip_by_value_2/MinimumMinimumlstm_cell_2/Add_7:z:0.lstm_cell_2/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:??????????b
lstm_cell_2/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
lstm_cell_2/clip_by_value_2Maximum'lstm_cell_2/clip_by_value_2/Minimum:z:0&lstm_cell_2/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:??????????d
lstm_cell_2/Tanh_1Tanhlstm_cell_2/add_5:z:0*
T0*(
_output_shapes
:???????????
lstm_cell_2/mul_5Mullstm_cell_2/clip_by_value_2:z:0lstm_cell_2/Tanh_1:y:0*
T0*(
_output_shapes
:??????????n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)lstm_cell_2_split_readvariableop_resource+lstm_cell_2_split_1_readvariableop_resource#lstm_cell_2_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_57949*
condR
while_cond_57948*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????c
IdentityIdentitytranspose_1:y:0^NoOp*
T0*,
_output_shapes
:???????????
NoOpNoOp^lstm_cell_2/ReadVariableOp^lstm_cell_2/ReadVariableOp_1^lstm_cell_2/ReadVariableOp_2^lstm_cell_2/ReadVariableOp_3!^lstm_cell_2/split/ReadVariableOp#^lstm_cell_2/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????: : : 28
lstm_cell_2/ReadVariableOplstm_cell_2/ReadVariableOp2<
lstm_cell_2/ReadVariableOp_1lstm_cell_2/ReadVariableOp_12<
lstm_cell_2/ReadVariableOp_2lstm_cell_2/ReadVariableOp_22<
lstm_cell_2/ReadVariableOp_3lstm_cell_2/ReadVariableOp_32D
 lstm_cell_2/split/ReadVariableOp lstm_cell_2/split/ReadVariableOp2H
"lstm_cell_2/split_1/ReadVariableOp"lstm_cell_2/split_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?
lstm_3_while_body_56595*
&lstm_3_while_lstm_3_while_loop_counter0
,lstm_3_while_lstm_3_while_maximum_iterations
lstm_3_while_placeholder
lstm_3_while_placeholder_1
lstm_3_while_placeholder_2
lstm_3_while_placeholder_3)
%lstm_3_while_lstm_3_strided_slice_1_0e
alstm_3_while_tensorarrayv2read_tensorlistgetitem_lstm_3_tensorarrayunstack_tensorlistfromtensor_0L
8lstm_3_while_lstm_cell_3_split_readvariableop_resource_0:
??I
:lstm_3_while_lstm_cell_3_split_1_readvariableop_resource_0:	?F
2lstm_3_while_lstm_cell_3_readvariableop_resource_0:
??
lstm_3_while_identity
lstm_3_while_identity_1
lstm_3_while_identity_2
lstm_3_while_identity_3
lstm_3_while_identity_4
lstm_3_while_identity_5'
#lstm_3_while_lstm_3_strided_slice_1c
_lstm_3_while_tensorarrayv2read_tensorlistgetitem_lstm_3_tensorarrayunstack_tensorlistfromtensorJ
6lstm_3_while_lstm_cell_3_split_readvariableop_resource:
??G
8lstm_3_while_lstm_cell_3_split_1_readvariableop_resource:	?D
0lstm_3_while_lstm_cell_3_readvariableop_resource:
????'lstm_3/while/lstm_cell_3/ReadVariableOp?)lstm_3/while/lstm_cell_3/ReadVariableOp_1?)lstm_3/while/lstm_cell_3/ReadVariableOp_2?)lstm_3/while/lstm_cell_3/ReadVariableOp_3?-lstm_3/while/lstm_cell_3/split/ReadVariableOp?/lstm_3/while/lstm_cell_3/split_1/ReadVariableOp?
>lstm_3/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
0lstm_3/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemalstm_3_while_tensorarrayv2read_tensorlistgetitem_lstm_3_tensorarrayunstack_tensorlistfromtensor_0lstm_3_while_placeholderGlstm_3/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype0j
(lstm_3/while/lstm_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
-lstm_3/while/lstm_cell_3/split/ReadVariableOpReadVariableOp8lstm_3_while_lstm_cell_3_split_readvariableop_resource_0* 
_output_shapes
:
??*
dtype0?
lstm_3/while/lstm_cell_3/splitSplit1lstm_3/while/lstm_cell_3/split/split_dim:output:05lstm_3/while/lstm_cell_3/split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
??:
??:
??:
??*
	num_split?
lstm_3/while/lstm_cell_3/MatMulMatMul7lstm_3/while/TensorArrayV2Read/TensorListGetItem:item:0'lstm_3/while/lstm_cell_3/split:output:0*
T0*(
_output_shapes
:???????????
!lstm_3/while/lstm_cell_3/MatMul_1MatMul7lstm_3/while/TensorArrayV2Read/TensorListGetItem:item:0'lstm_3/while/lstm_cell_3/split:output:1*
T0*(
_output_shapes
:???????????
!lstm_3/while/lstm_cell_3/MatMul_2MatMul7lstm_3/while/TensorArrayV2Read/TensorListGetItem:item:0'lstm_3/while/lstm_cell_3/split:output:2*
T0*(
_output_shapes
:???????????
!lstm_3/while/lstm_cell_3/MatMul_3MatMul7lstm_3/while/TensorArrayV2Read/TensorListGetItem:item:0'lstm_3/while/lstm_cell_3/split:output:3*
T0*(
_output_shapes
:??????????l
*lstm_3/while/lstm_cell_3/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
/lstm_3/while/lstm_cell_3/split_1/ReadVariableOpReadVariableOp:lstm_3_while_lstm_cell_3_split_1_readvariableop_resource_0*
_output_shapes	
:?*
dtype0?
 lstm_3/while/lstm_cell_3/split_1Split3lstm_3/while/lstm_cell_3/split_1/split_dim:output:07lstm_3/while/lstm_cell_3/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split?
 lstm_3/while/lstm_cell_3/BiasAddBiasAdd)lstm_3/while/lstm_cell_3/MatMul:product:0)lstm_3/while/lstm_cell_3/split_1:output:0*
T0*(
_output_shapes
:???????????
"lstm_3/while/lstm_cell_3/BiasAdd_1BiasAdd+lstm_3/while/lstm_cell_3/MatMul_1:product:0)lstm_3/while/lstm_cell_3/split_1:output:1*
T0*(
_output_shapes
:???????????
"lstm_3/while/lstm_cell_3/BiasAdd_2BiasAdd+lstm_3/while/lstm_cell_3/MatMul_2:product:0)lstm_3/while/lstm_cell_3/split_1:output:2*
T0*(
_output_shapes
:???????????
"lstm_3/while/lstm_cell_3/BiasAdd_3BiasAdd+lstm_3/while/lstm_cell_3/MatMul_3:product:0)lstm_3/while/lstm_cell_3/split_1:output:3*
T0*(
_output_shapes
:???????????
'lstm_3/while/lstm_cell_3/ReadVariableOpReadVariableOp2lstm_3_while_lstm_cell_3_readvariableop_resource_0* 
_output_shapes
:
??*
dtype0}
,lstm_3/while/lstm_cell_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        
.lstm_3/while/lstm_cell_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   
.lstm_3/while/lstm_cell_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
&lstm_3/while/lstm_cell_3/strided_sliceStridedSlice/lstm_3/while/lstm_cell_3/ReadVariableOp:value:05lstm_3/while/lstm_cell_3/strided_slice/stack:output:07lstm_3/while/lstm_cell_3/strided_slice/stack_1:output:07lstm_3/while/lstm_cell_3/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
!lstm_3/while/lstm_cell_3/MatMul_4MatMullstm_3_while_placeholder_2/lstm_3/while/lstm_cell_3/strided_slice:output:0*
T0*(
_output_shapes
:???????????
lstm_3/while/lstm_cell_3/addAddV2)lstm_3/while/lstm_cell_3/BiasAdd:output:0+lstm_3/while/lstm_cell_3/MatMul_4:product:0*
T0*(
_output_shapes
:??????????c
lstm_3/while/lstm_cell_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>e
 lstm_3/while/lstm_cell_3/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ??
lstm_3/while/lstm_cell_3/MulMul lstm_3/while/lstm_cell_3/add:z:0'lstm_3/while/lstm_cell_3/Const:output:0*
T0*(
_output_shapes
:???????????
lstm_3/while/lstm_cell_3/Add_1AddV2 lstm_3/while/lstm_cell_3/Mul:z:0)lstm_3/while/lstm_cell_3/Const_1:output:0*
T0*(
_output_shapes
:??????????u
0lstm_3/while/lstm_cell_3/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
.lstm_3/while/lstm_cell_3/clip_by_value/MinimumMinimum"lstm_3/while/lstm_cell_3/Add_1:z:09lstm_3/while/lstm_cell_3/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:??????????m
(lstm_3/while/lstm_cell_3/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
&lstm_3/while/lstm_cell_3/clip_by_valueMaximum2lstm_3/while/lstm_cell_3/clip_by_value/Minimum:z:01lstm_3/while/lstm_cell_3/clip_by_value/y:output:0*
T0*(
_output_shapes
:???????????
)lstm_3/while/lstm_cell_3/ReadVariableOp_1ReadVariableOp2lstm_3_while_lstm_cell_3_readvariableop_resource_0* 
_output_shapes
:
??*
dtype0
.lstm_3/while/lstm_cell_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   ?
0lstm_3/while/lstm_cell_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
0lstm_3/while/lstm_cell_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
(lstm_3/while/lstm_cell_3/strided_slice_1StridedSlice1lstm_3/while/lstm_cell_3/ReadVariableOp_1:value:07lstm_3/while/lstm_cell_3/strided_slice_1/stack:output:09lstm_3/while/lstm_cell_3/strided_slice_1/stack_1:output:09lstm_3/while/lstm_cell_3/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
!lstm_3/while/lstm_cell_3/MatMul_5MatMullstm_3_while_placeholder_21lstm_3/while/lstm_cell_3/strided_slice_1:output:0*
T0*(
_output_shapes
:???????????
lstm_3/while/lstm_cell_3/add_2AddV2+lstm_3/while/lstm_cell_3/BiasAdd_1:output:0+lstm_3/while/lstm_cell_3/MatMul_5:product:0*
T0*(
_output_shapes
:??????????e
 lstm_3/while/lstm_cell_3/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>e
 lstm_3/while/lstm_cell_3/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ??
lstm_3/while/lstm_cell_3/Mul_1Mul"lstm_3/while/lstm_cell_3/add_2:z:0)lstm_3/while/lstm_cell_3/Const_2:output:0*
T0*(
_output_shapes
:???????????
lstm_3/while/lstm_cell_3/Add_3AddV2"lstm_3/while/lstm_cell_3/Mul_1:z:0)lstm_3/while/lstm_cell_3/Const_3:output:0*
T0*(
_output_shapes
:??????????w
2lstm_3/while/lstm_cell_3/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
0lstm_3/while/lstm_cell_3/clip_by_value_1/MinimumMinimum"lstm_3/while/lstm_cell_3/Add_3:z:0;lstm_3/while/lstm_cell_3/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:??????????o
*lstm_3/while/lstm_cell_3/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
(lstm_3/while/lstm_cell_3/clip_by_value_1Maximum4lstm_3/while/lstm_cell_3/clip_by_value_1/Minimum:z:03lstm_3/while/lstm_cell_3/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:???????????
lstm_3/while/lstm_cell_3/mul_2Mul,lstm_3/while/lstm_cell_3/clip_by_value_1:z:0lstm_3_while_placeholder_3*
T0*(
_output_shapes
:???????????
)lstm_3/while/lstm_cell_3/ReadVariableOp_2ReadVariableOp2lstm_3_while_lstm_cell_3_readvariableop_resource_0* 
_output_shapes
:
??*
dtype0
.lstm_3/while/lstm_cell_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       ?
0lstm_3/while/lstm_cell_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  ?
0lstm_3/while/lstm_cell_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
(lstm_3/while/lstm_cell_3/strided_slice_2StridedSlice1lstm_3/while/lstm_cell_3/ReadVariableOp_2:value:07lstm_3/while/lstm_cell_3/strided_slice_2/stack:output:09lstm_3/while/lstm_cell_3/strided_slice_2/stack_1:output:09lstm_3/while/lstm_cell_3/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
!lstm_3/while/lstm_cell_3/MatMul_6MatMullstm_3_while_placeholder_21lstm_3/while/lstm_cell_3/strided_slice_2:output:0*
T0*(
_output_shapes
:???????????
lstm_3/while/lstm_cell_3/add_4AddV2+lstm_3/while/lstm_cell_3/BiasAdd_2:output:0+lstm_3/while/lstm_cell_3/MatMul_6:product:0*
T0*(
_output_shapes
:??????????|
lstm_3/while/lstm_cell_3/TanhTanh"lstm_3/while/lstm_cell_3/add_4:z:0*
T0*(
_output_shapes
:???????????
lstm_3/while/lstm_cell_3/mul_3Mul*lstm_3/while/lstm_cell_3/clip_by_value:z:0!lstm_3/while/lstm_cell_3/Tanh:y:0*
T0*(
_output_shapes
:???????????
lstm_3/while/lstm_cell_3/add_5AddV2"lstm_3/while/lstm_cell_3/mul_2:z:0"lstm_3/while/lstm_cell_3/mul_3:z:0*
T0*(
_output_shapes
:???????????
)lstm_3/while/lstm_cell_3/ReadVariableOp_3ReadVariableOp2lstm_3_while_lstm_cell_3_readvariableop_resource_0* 
_output_shapes
:
??*
dtype0
.lstm_3/while/lstm_cell_3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  ?
0lstm_3/while/lstm_cell_3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        ?
0lstm_3/while/lstm_cell_3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
(lstm_3/while/lstm_cell_3/strided_slice_3StridedSlice1lstm_3/while/lstm_cell_3/ReadVariableOp_3:value:07lstm_3/while/lstm_cell_3/strided_slice_3/stack:output:09lstm_3/while/lstm_cell_3/strided_slice_3/stack_1:output:09lstm_3/while/lstm_cell_3/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
!lstm_3/while/lstm_cell_3/MatMul_7MatMullstm_3_while_placeholder_21lstm_3/while/lstm_cell_3/strided_slice_3:output:0*
T0*(
_output_shapes
:???????????
lstm_3/while/lstm_cell_3/add_6AddV2+lstm_3/while/lstm_cell_3/BiasAdd_3:output:0+lstm_3/while/lstm_cell_3/MatMul_7:product:0*
T0*(
_output_shapes
:??????????e
 lstm_3/while/lstm_cell_3/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>e
 lstm_3/while/lstm_cell_3/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ??
lstm_3/while/lstm_cell_3/Mul_4Mul"lstm_3/while/lstm_cell_3/add_6:z:0)lstm_3/while/lstm_cell_3/Const_4:output:0*
T0*(
_output_shapes
:???????????
lstm_3/while/lstm_cell_3/Add_7AddV2"lstm_3/while/lstm_cell_3/Mul_4:z:0)lstm_3/while/lstm_cell_3/Const_5:output:0*
T0*(
_output_shapes
:??????????w
2lstm_3/while/lstm_cell_3/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
0lstm_3/while/lstm_cell_3/clip_by_value_2/MinimumMinimum"lstm_3/while/lstm_cell_3/Add_7:z:0;lstm_3/while/lstm_cell_3/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:??????????o
*lstm_3/while/lstm_cell_3/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
(lstm_3/while/lstm_cell_3/clip_by_value_2Maximum4lstm_3/while/lstm_cell_3/clip_by_value_2/Minimum:z:03lstm_3/while/lstm_cell_3/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:??????????~
lstm_3/while/lstm_cell_3/Tanh_1Tanh"lstm_3/while/lstm_cell_3/add_5:z:0*
T0*(
_output_shapes
:???????????
lstm_3/while/lstm_cell_3/mul_5Mul,lstm_3/while/lstm_cell_3/clip_by_value_2:z:0#lstm_3/while/lstm_cell_3/Tanh_1:y:0*
T0*(
_output_shapes
:???????????
1lstm_3/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_3_while_placeholder_1lstm_3_while_placeholder"lstm_3/while/lstm_cell_3/mul_5:z:0*
_output_shapes
: *
element_dtype0:???T
lstm_3/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :q
lstm_3/while/addAddV2lstm_3_while_placeholderlstm_3/while/add/y:output:0*
T0*
_output_shapes
: V
lstm_3/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :?
lstm_3/while/add_1AddV2&lstm_3_while_lstm_3_while_loop_counterlstm_3/while/add_1/y:output:0*
T0*
_output_shapes
: n
lstm_3/while/IdentityIdentitylstm_3/while/add_1:z:0^lstm_3/while/NoOp*
T0*
_output_shapes
: ?
lstm_3/while/Identity_1Identity,lstm_3_while_lstm_3_while_maximum_iterations^lstm_3/while/NoOp*
T0*
_output_shapes
: n
lstm_3/while/Identity_2Identitylstm_3/while/add:z:0^lstm_3/while/NoOp*
T0*
_output_shapes
: ?
lstm_3/while/Identity_3IdentityAlstm_3/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_3/while/NoOp*
T0*
_output_shapes
: ?
lstm_3/while/Identity_4Identity"lstm_3/while/lstm_cell_3/mul_5:z:0^lstm_3/while/NoOp*
T0*(
_output_shapes
:???????????
lstm_3/while/Identity_5Identity"lstm_3/while/lstm_cell_3/add_5:z:0^lstm_3/while/NoOp*
T0*(
_output_shapes
:???????????
lstm_3/while/NoOpNoOp(^lstm_3/while/lstm_cell_3/ReadVariableOp*^lstm_3/while/lstm_cell_3/ReadVariableOp_1*^lstm_3/while/lstm_cell_3/ReadVariableOp_2*^lstm_3/while/lstm_cell_3/ReadVariableOp_3.^lstm_3/while/lstm_cell_3/split/ReadVariableOp0^lstm_3/while/lstm_cell_3/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "7
lstm_3_while_identitylstm_3/while/Identity:output:0";
lstm_3_while_identity_1 lstm_3/while/Identity_1:output:0";
lstm_3_while_identity_2 lstm_3/while/Identity_2:output:0";
lstm_3_while_identity_3 lstm_3/while/Identity_3:output:0";
lstm_3_while_identity_4 lstm_3/while/Identity_4:output:0";
lstm_3_while_identity_5 lstm_3/while/Identity_5:output:0"L
#lstm_3_while_lstm_3_strided_slice_1%lstm_3_while_lstm_3_strided_slice_1_0"f
0lstm_3_while_lstm_cell_3_readvariableop_resource2lstm_3_while_lstm_cell_3_readvariableop_resource_0"v
8lstm_3_while_lstm_cell_3_split_1_readvariableop_resource:lstm_3_while_lstm_cell_3_split_1_readvariableop_resource_0"r
6lstm_3_while_lstm_cell_3_split_readvariableop_resource8lstm_3_while_lstm_cell_3_split_readvariableop_resource_0"?
_lstm_3_while_tensorarrayv2read_tensorlistgetitem_lstm_3_tensorarrayunstack_tensorlistfromtensoralstm_3_while_tensorarrayv2read_tensorlistgetitem_lstm_3_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2R
'lstm_3/while/lstm_cell_3/ReadVariableOp'lstm_3/while/lstm_cell_3/ReadVariableOp2V
)lstm_3/while/lstm_cell_3/ReadVariableOp_1)lstm_3/while/lstm_cell_3/ReadVariableOp_12V
)lstm_3/while/lstm_cell_3/ReadVariableOp_2)lstm_3/while/lstm_cell_3/ReadVariableOp_22V
)lstm_3/while/lstm_cell_3/ReadVariableOp_3)lstm_3/while/lstm_cell_3/ReadVariableOp_32^
-lstm_3/while/lstm_cell_3/split/ReadVariableOp-lstm_3/while/lstm_cell_3/split/ReadVariableOp2b
/lstm_3/while/lstm_cell_3/split_1/ReadVariableOp/lstm_3/while/lstm_cell_3/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
??
?
$sequential_1_lstm_3_while_body_53790D
@sequential_1_lstm_3_while_sequential_1_lstm_3_while_loop_counterJ
Fsequential_1_lstm_3_while_sequential_1_lstm_3_while_maximum_iterations)
%sequential_1_lstm_3_while_placeholder+
'sequential_1_lstm_3_while_placeholder_1+
'sequential_1_lstm_3_while_placeholder_2+
'sequential_1_lstm_3_while_placeholder_3C
?sequential_1_lstm_3_while_sequential_1_lstm_3_strided_slice_1_0
{sequential_1_lstm_3_while_tensorarrayv2read_tensorlistgetitem_sequential_1_lstm_3_tensorarrayunstack_tensorlistfromtensor_0Y
Esequential_1_lstm_3_while_lstm_cell_3_split_readvariableop_resource_0:
??V
Gsequential_1_lstm_3_while_lstm_cell_3_split_1_readvariableop_resource_0:	?S
?sequential_1_lstm_3_while_lstm_cell_3_readvariableop_resource_0:
??&
"sequential_1_lstm_3_while_identity(
$sequential_1_lstm_3_while_identity_1(
$sequential_1_lstm_3_while_identity_2(
$sequential_1_lstm_3_while_identity_3(
$sequential_1_lstm_3_while_identity_4(
$sequential_1_lstm_3_while_identity_5A
=sequential_1_lstm_3_while_sequential_1_lstm_3_strided_slice_1}
ysequential_1_lstm_3_while_tensorarrayv2read_tensorlistgetitem_sequential_1_lstm_3_tensorarrayunstack_tensorlistfromtensorW
Csequential_1_lstm_3_while_lstm_cell_3_split_readvariableop_resource:
??T
Esequential_1_lstm_3_while_lstm_cell_3_split_1_readvariableop_resource:	?Q
=sequential_1_lstm_3_while_lstm_cell_3_readvariableop_resource:
????4sequential_1/lstm_3/while/lstm_cell_3/ReadVariableOp?6sequential_1/lstm_3/while/lstm_cell_3/ReadVariableOp_1?6sequential_1/lstm_3/while/lstm_cell_3/ReadVariableOp_2?6sequential_1/lstm_3/while/lstm_cell_3/ReadVariableOp_3?:sequential_1/lstm_3/while/lstm_cell_3/split/ReadVariableOp?<sequential_1/lstm_3/while/lstm_cell_3/split_1/ReadVariableOp?
Ksequential_1/lstm_3/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
=sequential_1/lstm_3/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem{sequential_1_lstm_3_while_tensorarrayv2read_tensorlistgetitem_sequential_1_lstm_3_tensorarrayunstack_tensorlistfromtensor_0%sequential_1_lstm_3_while_placeholderTsequential_1/lstm_3/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype0w
5sequential_1/lstm_3/while/lstm_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
:sequential_1/lstm_3/while/lstm_cell_3/split/ReadVariableOpReadVariableOpEsequential_1_lstm_3_while_lstm_cell_3_split_readvariableop_resource_0* 
_output_shapes
:
??*
dtype0?
+sequential_1/lstm_3/while/lstm_cell_3/splitSplit>sequential_1/lstm_3/while/lstm_cell_3/split/split_dim:output:0Bsequential_1/lstm_3/while/lstm_cell_3/split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
??:
??:
??:
??*
	num_split?
,sequential_1/lstm_3/while/lstm_cell_3/MatMulMatMulDsequential_1/lstm_3/while/TensorArrayV2Read/TensorListGetItem:item:04sequential_1/lstm_3/while/lstm_cell_3/split:output:0*
T0*(
_output_shapes
:???????????
.sequential_1/lstm_3/while/lstm_cell_3/MatMul_1MatMulDsequential_1/lstm_3/while/TensorArrayV2Read/TensorListGetItem:item:04sequential_1/lstm_3/while/lstm_cell_3/split:output:1*
T0*(
_output_shapes
:???????????
.sequential_1/lstm_3/while/lstm_cell_3/MatMul_2MatMulDsequential_1/lstm_3/while/TensorArrayV2Read/TensorListGetItem:item:04sequential_1/lstm_3/while/lstm_cell_3/split:output:2*
T0*(
_output_shapes
:???????????
.sequential_1/lstm_3/while/lstm_cell_3/MatMul_3MatMulDsequential_1/lstm_3/while/TensorArrayV2Read/TensorListGetItem:item:04sequential_1/lstm_3/while/lstm_cell_3/split:output:3*
T0*(
_output_shapes
:??????????y
7sequential_1/lstm_3/while/lstm_cell_3/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
<sequential_1/lstm_3/while/lstm_cell_3/split_1/ReadVariableOpReadVariableOpGsequential_1_lstm_3_while_lstm_cell_3_split_1_readvariableop_resource_0*
_output_shapes	
:?*
dtype0?
-sequential_1/lstm_3/while/lstm_cell_3/split_1Split@sequential_1/lstm_3/while/lstm_cell_3/split_1/split_dim:output:0Dsequential_1/lstm_3/while/lstm_cell_3/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split?
-sequential_1/lstm_3/while/lstm_cell_3/BiasAddBiasAdd6sequential_1/lstm_3/while/lstm_cell_3/MatMul:product:06sequential_1/lstm_3/while/lstm_cell_3/split_1:output:0*
T0*(
_output_shapes
:???????????
/sequential_1/lstm_3/while/lstm_cell_3/BiasAdd_1BiasAdd8sequential_1/lstm_3/while/lstm_cell_3/MatMul_1:product:06sequential_1/lstm_3/while/lstm_cell_3/split_1:output:1*
T0*(
_output_shapes
:???????????
/sequential_1/lstm_3/while/lstm_cell_3/BiasAdd_2BiasAdd8sequential_1/lstm_3/while/lstm_cell_3/MatMul_2:product:06sequential_1/lstm_3/while/lstm_cell_3/split_1:output:2*
T0*(
_output_shapes
:???????????
/sequential_1/lstm_3/while/lstm_cell_3/BiasAdd_3BiasAdd8sequential_1/lstm_3/while/lstm_cell_3/MatMul_3:product:06sequential_1/lstm_3/while/lstm_cell_3/split_1:output:3*
T0*(
_output_shapes
:???????????
4sequential_1/lstm_3/while/lstm_cell_3/ReadVariableOpReadVariableOp?sequential_1_lstm_3_while_lstm_cell_3_readvariableop_resource_0* 
_output_shapes
:
??*
dtype0?
9sequential_1/lstm_3/while/lstm_cell_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
;sequential_1/lstm_3/while/lstm_cell_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   ?
;sequential_1/lstm_3/while/lstm_cell_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
3sequential_1/lstm_3/while/lstm_cell_3/strided_sliceStridedSlice<sequential_1/lstm_3/while/lstm_cell_3/ReadVariableOp:value:0Bsequential_1/lstm_3/while/lstm_cell_3/strided_slice/stack:output:0Dsequential_1/lstm_3/while/lstm_cell_3/strided_slice/stack_1:output:0Dsequential_1/lstm_3/while/lstm_cell_3/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
.sequential_1/lstm_3/while/lstm_cell_3/MatMul_4MatMul'sequential_1_lstm_3_while_placeholder_2<sequential_1/lstm_3/while/lstm_cell_3/strided_slice:output:0*
T0*(
_output_shapes
:???????????
)sequential_1/lstm_3/while/lstm_cell_3/addAddV26sequential_1/lstm_3/while/lstm_cell_3/BiasAdd:output:08sequential_1/lstm_3/while/lstm_cell_3/MatMul_4:product:0*
T0*(
_output_shapes
:??????????p
+sequential_1/lstm_3/while/lstm_cell_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>r
-sequential_1/lstm_3/while/lstm_cell_3/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ??
)sequential_1/lstm_3/while/lstm_cell_3/MulMul-sequential_1/lstm_3/while/lstm_cell_3/add:z:04sequential_1/lstm_3/while/lstm_cell_3/Const:output:0*
T0*(
_output_shapes
:???????????
+sequential_1/lstm_3/while/lstm_cell_3/Add_1AddV2-sequential_1/lstm_3/while/lstm_cell_3/Mul:z:06sequential_1/lstm_3/while/lstm_cell_3/Const_1:output:0*
T0*(
_output_shapes
:???????????
=sequential_1/lstm_3/while/lstm_cell_3/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
;sequential_1/lstm_3/while/lstm_cell_3/clip_by_value/MinimumMinimum/sequential_1/lstm_3/while/lstm_cell_3/Add_1:z:0Fsequential_1/lstm_3/while/lstm_cell_3/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:??????????z
5sequential_1/lstm_3/while/lstm_cell_3/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
3sequential_1/lstm_3/while/lstm_cell_3/clip_by_valueMaximum?sequential_1/lstm_3/while/lstm_cell_3/clip_by_value/Minimum:z:0>sequential_1/lstm_3/while/lstm_cell_3/clip_by_value/y:output:0*
T0*(
_output_shapes
:???????????
6sequential_1/lstm_3/while/lstm_cell_3/ReadVariableOp_1ReadVariableOp?sequential_1_lstm_3_while_lstm_cell_3_readvariableop_resource_0* 
_output_shapes
:
??*
dtype0?
;sequential_1/lstm_3/while/lstm_cell_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   ?
=sequential_1/lstm_3/while/lstm_cell_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
=sequential_1/lstm_3/while/lstm_cell_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
5sequential_1/lstm_3/while/lstm_cell_3/strided_slice_1StridedSlice>sequential_1/lstm_3/while/lstm_cell_3/ReadVariableOp_1:value:0Dsequential_1/lstm_3/while/lstm_cell_3/strided_slice_1/stack:output:0Fsequential_1/lstm_3/while/lstm_cell_3/strided_slice_1/stack_1:output:0Fsequential_1/lstm_3/while/lstm_cell_3/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
.sequential_1/lstm_3/while/lstm_cell_3/MatMul_5MatMul'sequential_1_lstm_3_while_placeholder_2>sequential_1/lstm_3/while/lstm_cell_3/strided_slice_1:output:0*
T0*(
_output_shapes
:???????????
+sequential_1/lstm_3/while/lstm_cell_3/add_2AddV28sequential_1/lstm_3/while/lstm_cell_3/BiasAdd_1:output:08sequential_1/lstm_3/while/lstm_cell_3/MatMul_5:product:0*
T0*(
_output_shapes
:??????????r
-sequential_1/lstm_3/while/lstm_cell_3/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>r
-sequential_1/lstm_3/while/lstm_cell_3/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ??
+sequential_1/lstm_3/while/lstm_cell_3/Mul_1Mul/sequential_1/lstm_3/while/lstm_cell_3/add_2:z:06sequential_1/lstm_3/while/lstm_cell_3/Const_2:output:0*
T0*(
_output_shapes
:???????????
+sequential_1/lstm_3/while/lstm_cell_3/Add_3AddV2/sequential_1/lstm_3/while/lstm_cell_3/Mul_1:z:06sequential_1/lstm_3/while/lstm_cell_3/Const_3:output:0*
T0*(
_output_shapes
:???????????
?sequential_1/lstm_3/while/lstm_cell_3/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
=sequential_1/lstm_3/while/lstm_cell_3/clip_by_value_1/MinimumMinimum/sequential_1/lstm_3/while/lstm_cell_3/Add_3:z:0Hsequential_1/lstm_3/while/lstm_cell_3/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:??????????|
7sequential_1/lstm_3/while/lstm_cell_3/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
5sequential_1/lstm_3/while/lstm_cell_3/clip_by_value_1MaximumAsequential_1/lstm_3/while/lstm_cell_3/clip_by_value_1/Minimum:z:0@sequential_1/lstm_3/while/lstm_cell_3/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:???????????
+sequential_1/lstm_3/while/lstm_cell_3/mul_2Mul9sequential_1/lstm_3/while/lstm_cell_3/clip_by_value_1:z:0'sequential_1_lstm_3_while_placeholder_3*
T0*(
_output_shapes
:???????????
6sequential_1/lstm_3/while/lstm_cell_3/ReadVariableOp_2ReadVariableOp?sequential_1_lstm_3_while_lstm_cell_3_readvariableop_resource_0* 
_output_shapes
:
??*
dtype0?
;sequential_1/lstm_3/while/lstm_cell_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       ?
=sequential_1/lstm_3/while/lstm_cell_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  ?
=sequential_1/lstm_3/while/lstm_cell_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
5sequential_1/lstm_3/while/lstm_cell_3/strided_slice_2StridedSlice>sequential_1/lstm_3/while/lstm_cell_3/ReadVariableOp_2:value:0Dsequential_1/lstm_3/while/lstm_cell_3/strided_slice_2/stack:output:0Fsequential_1/lstm_3/while/lstm_cell_3/strided_slice_2/stack_1:output:0Fsequential_1/lstm_3/while/lstm_cell_3/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
.sequential_1/lstm_3/while/lstm_cell_3/MatMul_6MatMul'sequential_1_lstm_3_while_placeholder_2>sequential_1/lstm_3/while/lstm_cell_3/strided_slice_2:output:0*
T0*(
_output_shapes
:???????????
+sequential_1/lstm_3/while/lstm_cell_3/add_4AddV28sequential_1/lstm_3/while/lstm_cell_3/BiasAdd_2:output:08sequential_1/lstm_3/while/lstm_cell_3/MatMul_6:product:0*
T0*(
_output_shapes
:???????????
*sequential_1/lstm_3/while/lstm_cell_3/TanhTanh/sequential_1/lstm_3/while/lstm_cell_3/add_4:z:0*
T0*(
_output_shapes
:???????????
+sequential_1/lstm_3/while/lstm_cell_3/mul_3Mul7sequential_1/lstm_3/while/lstm_cell_3/clip_by_value:z:0.sequential_1/lstm_3/while/lstm_cell_3/Tanh:y:0*
T0*(
_output_shapes
:???????????
+sequential_1/lstm_3/while/lstm_cell_3/add_5AddV2/sequential_1/lstm_3/while/lstm_cell_3/mul_2:z:0/sequential_1/lstm_3/while/lstm_cell_3/mul_3:z:0*
T0*(
_output_shapes
:???????????
6sequential_1/lstm_3/while/lstm_cell_3/ReadVariableOp_3ReadVariableOp?sequential_1_lstm_3_while_lstm_cell_3_readvariableop_resource_0* 
_output_shapes
:
??*
dtype0?
;sequential_1/lstm_3/while/lstm_cell_3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  ?
=sequential_1/lstm_3/while/lstm_cell_3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        ?
=sequential_1/lstm_3/while/lstm_cell_3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
5sequential_1/lstm_3/while/lstm_cell_3/strided_slice_3StridedSlice>sequential_1/lstm_3/while/lstm_cell_3/ReadVariableOp_3:value:0Dsequential_1/lstm_3/while/lstm_cell_3/strided_slice_3/stack:output:0Fsequential_1/lstm_3/while/lstm_cell_3/strided_slice_3/stack_1:output:0Fsequential_1/lstm_3/while/lstm_cell_3/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
.sequential_1/lstm_3/while/lstm_cell_3/MatMul_7MatMul'sequential_1_lstm_3_while_placeholder_2>sequential_1/lstm_3/while/lstm_cell_3/strided_slice_3:output:0*
T0*(
_output_shapes
:???????????
+sequential_1/lstm_3/while/lstm_cell_3/add_6AddV28sequential_1/lstm_3/while/lstm_cell_3/BiasAdd_3:output:08sequential_1/lstm_3/while/lstm_cell_3/MatMul_7:product:0*
T0*(
_output_shapes
:??????????r
-sequential_1/lstm_3/while/lstm_cell_3/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>r
-sequential_1/lstm_3/while/lstm_cell_3/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ??
+sequential_1/lstm_3/while/lstm_cell_3/Mul_4Mul/sequential_1/lstm_3/while/lstm_cell_3/add_6:z:06sequential_1/lstm_3/while/lstm_cell_3/Const_4:output:0*
T0*(
_output_shapes
:???????????
+sequential_1/lstm_3/while/lstm_cell_3/Add_7AddV2/sequential_1/lstm_3/while/lstm_cell_3/Mul_4:z:06sequential_1/lstm_3/while/lstm_cell_3/Const_5:output:0*
T0*(
_output_shapes
:???????????
?sequential_1/lstm_3/while/lstm_cell_3/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
=sequential_1/lstm_3/while/lstm_cell_3/clip_by_value_2/MinimumMinimum/sequential_1/lstm_3/while/lstm_cell_3/Add_7:z:0Hsequential_1/lstm_3/while/lstm_cell_3/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:??????????|
7sequential_1/lstm_3/while/lstm_cell_3/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
5sequential_1/lstm_3/while/lstm_cell_3/clip_by_value_2MaximumAsequential_1/lstm_3/while/lstm_cell_3/clip_by_value_2/Minimum:z:0@sequential_1/lstm_3/while/lstm_cell_3/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:???????????
,sequential_1/lstm_3/while/lstm_cell_3/Tanh_1Tanh/sequential_1/lstm_3/while/lstm_cell_3/add_5:z:0*
T0*(
_output_shapes
:???????????
+sequential_1/lstm_3/while/lstm_cell_3/mul_5Mul9sequential_1/lstm_3/while/lstm_cell_3/clip_by_value_2:z:00sequential_1/lstm_3/while/lstm_cell_3/Tanh_1:y:0*
T0*(
_output_shapes
:???????????
>sequential_1/lstm_3/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem'sequential_1_lstm_3_while_placeholder_1%sequential_1_lstm_3_while_placeholder/sequential_1/lstm_3/while/lstm_cell_3/mul_5:z:0*
_output_shapes
: *
element_dtype0:???a
sequential_1/lstm_3/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
sequential_1/lstm_3/while/addAddV2%sequential_1_lstm_3_while_placeholder(sequential_1/lstm_3/while/add/y:output:0*
T0*
_output_shapes
: c
!sequential_1/lstm_3/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :?
sequential_1/lstm_3/while/add_1AddV2@sequential_1_lstm_3_while_sequential_1_lstm_3_while_loop_counter*sequential_1/lstm_3/while/add_1/y:output:0*
T0*
_output_shapes
: ?
"sequential_1/lstm_3/while/IdentityIdentity#sequential_1/lstm_3/while/add_1:z:0^sequential_1/lstm_3/while/NoOp*
T0*
_output_shapes
: ?
$sequential_1/lstm_3/while/Identity_1IdentityFsequential_1_lstm_3_while_sequential_1_lstm_3_while_maximum_iterations^sequential_1/lstm_3/while/NoOp*
T0*
_output_shapes
: ?
$sequential_1/lstm_3/while/Identity_2Identity!sequential_1/lstm_3/while/add:z:0^sequential_1/lstm_3/while/NoOp*
T0*
_output_shapes
: ?
$sequential_1/lstm_3/while/Identity_3IdentityNsequential_1/lstm_3/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^sequential_1/lstm_3/while/NoOp*
T0*
_output_shapes
: ?
$sequential_1/lstm_3/while/Identity_4Identity/sequential_1/lstm_3/while/lstm_cell_3/mul_5:z:0^sequential_1/lstm_3/while/NoOp*
T0*(
_output_shapes
:???????????
$sequential_1/lstm_3/while/Identity_5Identity/sequential_1/lstm_3/while/lstm_cell_3/add_5:z:0^sequential_1/lstm_3/while/NoOp*
T0*(
_output_shapes
:???????????
sequential_1/lstm_3/while/NoOpNoOp5^sequential_1/lstm_3/while/lstm_cell_3/ReadVariableOp7^sequential_1/lstm_3/while/lstm_cell_3/ReadVariableOp_17^sequential_1/lstm_3/while/lstm_cell_3/ReadVariableOp_27^sequential_1/lstm_3/while/lstm_cell_3/ReadVariableOp_3;^sequential_1/lstm_3/while/lstm_cell_3/split/ReadVariableOp=^sequential_1/lstm_3/while/lstm_cell_3/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "Q
"sequential_1_lstm_3_while_identity+sequential_1/lstm_3/while/Identity:output:0"U
$sequential_1_lstm_3_while_identity_1-sequential_1/lstm_3/while/Identity_1:output:0"U
$sequential_1_lstm_3_while_identity_2-sequential_1/lstm_3/while/Identity_2:output:0"U
$sequential_1_lstm_3_while_identity_3-sequential_1/lstm_3/while/Identity_3:output:0"U
$sequential_1_lstm_3_while_identity_4-sequential_1/lstm_3/while/Identity_4:output:0"U
$sequential_1_lstm_3_while_identity_5-sequential_1/lstm_3/while/Identity_5:output:0"?
=sequential_1_lstm_3_while_lstm_cell_3_readvariableop_resource?sequential_1_lstm_3_while_lstm_cell_3_readvariableop_resource_0"?
Esequential_1_lstm_3_while_lstm_cell_3_split_1_readvariableop_resourceGsequential_1_lstm_3_while_lstm_cell_3_split_1_readvariableop_resource_0"?
Csequential_1_lstm_3_while_lstm_cell_3_split_readvariableop_resourceEsequential_1_lstm_3_while_lstm_cell_3_split_readvariableop_resource_0"?
=sequential_1_lstm_3_while_sequential_1_lstm_3_strided_slice_1?sequential_1_lstm_3_while_sequential_1_lstm_3_strided_slice_1_0"?
ysequential_1_lstm_3_while_tensorarrayv2read_tensorlistgetitem_sequential_1_lstm_3_tensorarrayunstack_tensorlistfromtensor{sequential_1_lstm_3_while_tensorarrayv2read_tensorlistgetitem_sequential_1_lstm_3_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2l
4sequential_1/lstm_3/while/lstm_cell_3/ReadVariableOp4sequential_1/lstm_3/while/lstm_cell_3/ReadVariableOp2p
6sequential_1/lstm_3/while/lstm_cell_3/ReadVariableOp_16sequential_1/lstm_3/while/lstm_cell_3/ReadVariableOp_12p
6sequential_1/lstm_3/while/lstm_cell_3/ReadVariableOp_26sequential_1/lstm_3/while/lstm_cell_3/ReadVariableOp_22p
6sequential_1/lstm_3/while/lstm_cell_3/ReadVariableOp_36sequential_1/lstm_3/while/lstm_cell_3/ReadVariableOp_32x
:sequential_1/lstm_3/while/lstm_cell_3/split/ReadVariableOp:sequential_1/lstm_3/while/lstm_cell_3/split/ReadVariableOp2|
<sequential_1/lstm_3/while/lstm_cell_3/split_1/ReadVariableOp<sequential_1/lstm_3/while/lstm_cell_3/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
??
?
lstm_2_while_body_56865*
&lstm_2_while_lstm_2_while_loop_counter0
,lstm_2_while_lstm_2_while_maximum_iterations
lstm_2_while_placeholder
lstm_2_while_placeholder_1
lstm_2_while_placeholder_2
lstm_2_while_placeholder_3)
%lstm_2_while_lstm_2_strided_slice_1_0e
alstm_2_while_tensorarrayv2read_tensorlistgetitem_lstm_2_tensorarrayunstack_tensorlistfromtensor_0K
8lstm_2_while_lstm_cell_2_split_readvariableop_resource_0:	?I
:lstm_2_while_lstm_cell_2_split_1_readvariableop_resource_0:	?F
2lstm_2_while_lstm_cell_2_readvariableop_resource_0:
??
lstm_2_while_identity
lstm_2_while_identity_1
lstm_2_while_identity_2
lstm_2_while_identity_3
lstm_2_while_identity_4
lstm_2_while_identity_5'
#lstm_2_while_lstm_2_strided_slice_1c
_lstm_2_while_tensorarrayv2read_tensorlistgetitem_lstm_2_tensorarrayunstack_tensorlistfromtensorI
6lstm_2_while_lstm_cell_2_split_readvariableop_resource:	?G
8lstm_2_while_lstm_cell_2_split_1_readvariableop_resource:	?D
0lstm_2_while_lstm_cell_2_readvariableop_resource:
????'lstm_2/while/lstm_cell_2/ReadVariableOp?)lstm_2/while/lstm_cell_2/ReadVariableOp_1?)lstm_2/while/lstm_cell_2/ReadVariableOp_2?)lstm_2/while/lstm_cell_2/ReadVariableOp_3?-lstm_2/while/lstm_cell_2/split/ReadVariableOp?/lstm_2/while/lstm_cell_2/split_1/ReadVariableOp?
>lstm_2/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
0lstm_2/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemalstm_2_while_tensorarrayv2read_tensorlistgetitem_lstm_2_tensorarrayunstack_tensorlistfromtensor_0lstm_2_while_placeholderGlstm_2/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0j
(lstm_2/while/lstm_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
-lstm_2/while/lstm_cell_2/split/ReadVariableOpReadVariableOp8lstm_2_while_lstm_cell_2_split_readvariableop_resource_0*
_output_shapes
:	?*
dtype0?
lstm_2/while/lstm_cell_2/splitSplit1lstm_2/while/lstm_cell_2/split/split_dim:output:05lstm_2/while/lstm_cell_2/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	?:	?:	?:	?*
	num_split?
lstm_2/while/lstm_cell_2/MatMulMatMul7lstm_2/while/TensorArrayV2Read/TensorListGetItem:item:0'lstm_2/while/lstm_cell_2/split:output:0*
T0*(
_output_shapes
:???????????
!lstm_2/while/lstm_cell_2/MatMul_1MatMul7lstm_2/while/TensorArrayV2Read/TensorListGetItem:item:0'lstm_2/while/lstm_cell_2/split:output:1*
T0*(
_output_shapes
:???????????
!lstm_2/while/lstm_cell_2/MatMul_2MatMul7lstm_2/while/TensorArrayV2Read/TensorListGetItem:item:0'lstm_2/while/lstm_cell_2/split:output:2*
T0*(
_output_shapes
:???????????
!lstm_2/while/lstm_cell_2/MatMul_3MatMul7lstm_2/while/TensorArrayV2Read/TensorListGetItem:item:0'lstm_2/while/lstm_cell_2/split:output:3*
T0*(
_output_shapes
:??????????l
*lstm_2/while/lstm_cell_2/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
/lstm_2/while/lstm_cell_2/split_1/ReadVariableOpReadVariableOp:lstm_2_while_lstm_cell_2_split_1_readvariableop_resource_0*
_output_shapes	
:?*
dtype0?
 lstm_2/while/lstm_cell_2/split_1Split3lstm_2/while/lstm_cell_2/split_1/split_dim:output:07lstm_2/while/lstm_cell_2/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split?
 lstm_2/while/lstm_cell_2/BiasAddBiasAdd)lstm_2/while/lstm_cell_2/MatMul:product:0)lstm_2/while/lstm_cell_2/split_1:output:0*
T0*(
_output_shapes
:???????????
"lstm_2/while/lstm_cell_2/BiasAdd_1BiasAdd+lstm_2/while/lstm_cell_2/MatMul_1:product:0)lstm_2/while/lstm_cell_2/split_1:output:1*
T0*(
_output_shapes
:???????????
"lstm_2/while/lstm_cell_2/BiasAdd_2BiasAdd+lstm_2/while/lstm_cell_2/MatMul_2:product:0)lstm_2/while/lstm_cell_2/split_1:output:2*
T0*(
_output_shapes
:???????????
"lstm_2/while/lstm_cell_2/BiasAdd_3BiasAdd+lstm_2/while/lstm_cell_2/MatMul_3:product:0)lstm_2/while/lstm_cell_2/split_1:output:3*
T0*(
_output_shapes
:???????????
'lstm_2/while/lstm_cell_2/ReadVariableOpReadVariableOp2lstm_2_while_lstm_cell_2_readvariableop_resource_0* 
_output_shapes
:
??*
dtype0}
,lstm_2/while/lstm_cell_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        
.lstm_2/while/lstm_cell_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
.lstm_2/while/lstm_cell_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
&lstm_2/while/lstm_cell_2/strided_sliceStridedSlice/lstm_2/while/lstm_cell_2/ReadVariableOp:value:05lstm_2/while/lstm_cell_2/strided_slice/stack:output:07lstm_2/while/lstm_cell_2/strided_slice/stack_1:output:07lstm_2/while/lstm_cell_2/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
!lstm_2/while/lstm_cell_2/MatMul_4MatMullstm_2_while_placeholder_2/lstm_2/while/lstm_cell_2/strided_slice:output:0*
T0*(
_output_shapes
:???????????
lstm_2/while/lstm_cell_2/addAddV2)lstm_2/while/lstm_cell_2/BiasAdd:output:0+lstm_2/while/lstm_cell_2/MatMul_4:product:0*
T0*(
_output_shapes
:??????????c
lstm_2/while/lstm_cell_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>e
 lstm_2/while/lstm_cell_2/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ??
lstm_2/while/lstm_cell_2/MulMul lstm_2/while/lstm_cell_2/add:z:0'lstm_2/while/lstm_cell_2/Const:output:0*
T0*(
_output_shapes
:???????????
lstm_2/while/lstm_cell_2/Add_1AddV2 lstm_2/while/lstm_cell_2/Mul:z:0)lstm_2/while/lstm_cell_2/Const_1:output:0*
T0*(
_output_shapes
:??????????u
0lstm_2/while/lstm_cell_2/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
.lstm_2/while/lstm_cell_2/clip_by_value/MinimumMinimum"lstm_2/while/lstm_cell_2/Add_1:z:09lstm_2/while/lstm_cell_2/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:??????????m
(lstm_2/while/lstm_cell_2/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
&lstm_2/while/lstm_cell_2/clip_by_valueMaximum2lstm_2/while/lstm_cell_2/clip_by_value/Minimum:z:01lstm_2/while/lstm_cell_2/clip_by_value/y:output:0*
T0*(
_output_shapes
:???????????
)lstm_2/while/lstm_cell_2/ReadVariableOp_1ReadVariableOp2lstm_2_while_lstm_cell_2_readvariableop_resource_0* 
_output_shapes
:
??*
dtype0
.lstm_2/while/lstm_cell_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       ?
0lstm_2/while/lstm_cell_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
0lstm_2/while/lstm_cell_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
(lstm_2/while/lstm_cell_2/strided_slice_1StridedSlice1lstm_2/while/lstm_cell_2/ReadVariableOp_1:value:07lstm_2/while/lstm_cell_2/strided_slice_1/stack:output:09lstm_2/while/lstm_cell_2/strided_slice_1/stack_1:output:09lstm_2/while/lstm_cell_2/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
!lstm_2/while/lstm_cell_2/MatMul_5MatMullstm_2_while_placeholder_21lstm_2/while/lstm_cell_2/strided_slice_1:output:0*
T0*(
_output_shapes
:???????????
lstm_2/while/lstm_cell_2/add_2AddV2+lstm_2/while/lstm_cell_2/BiasAdd_1:output:0+lstm_2/while/lstm_cell_2/MatMul_5:product:0*
T0*(
_output_shapes
:??????????e
 lstm_2/while/lstm_cell_2/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>e
 lstm_2/while/lstm_cell_2/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ??
lstm_2/while/lstm_cell_2/Mul_1Mul"lstm_2/while/lstm_cell_2/add_2:z:0)lstm_2/while/lstm_cell_2/Const_2:output:0*
T0*(
_output_shapes
:???????????
lstm_2/while/lstm_cell_2/Add_3AddV2"lstm_2/while/lstm_cell_2/Mul_1:z:0)lstm_2/while/lstm_cell_2/Const_3:output:0*
T0*(
_output_shapes
:??????????w
2lstm_2/while/lstm_cell_2/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
0lstm_2/while/lstm_cell_2/clip_by_value_1/MinimumMinimum"lstm_2/while/lstm_cell_2/Add_3:z:0;lstm_2/while/lstm_cell_2/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:??????????o
*lstm_2/while/lstm_cell_2/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
(lstm_2/while/lstm_cell_2/clip_by_value_1Maximum4lstm_2/while/lstm_cell_2/clip_by_value_1/Minimum:z:03lstm_2/while/lstm_cell_2/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:???????????
lstm_2/while/lstm_cell_2/mul_2Mul,lstm_2/while/lstm_cell_2/clip_by_value_1:z:0lstm_2_while_placeholder_3*
T0*(
_output_shapes
:???????????
)lstm_2/while/lstm_cell_2/ReadVariableOp_2ReadVariableOp2lstm_2_while_lstm_cell_2_readvariableop_resource_0* 
_output_shapes
:
??*
dtype0
.lstm_2/while/lstm_cell_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       ?
0lstm_2/while/lstm_cell_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
0lstm_2/while/lstm_cell_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
(lstm_2/while/lstm_cell_2/strided_slice_2StridedSlice1lstm_2/while/lstm_cell_2/ReadVariableOp_2:value:07lstm_2/while/lstm_cell_2/strided_slice_2/stack:output:09lstm_2/while/lstm_cell_2/strided_slice_2/stack_1:output:09lstm_2/while/lstm_cell_2/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
!lstm_2/while/lstm_cell_2/MatMul_6MatMullstm_2_while_placeholder_21lstm_2/while/lstm_cell_2/strided_slice_2:output:0*
T0*(
_output_shapes
:???????????
lstm_2/while/lstm_cell_2/add_4AddV2+lstm_2/while/lstm_cell_2/BiasAdd_2:output:0+lstm_2/while/lstm_cell_2/MatMul_6:product:0*
T0*(
_output_shapes
:??????????|
lstm_2/while/lstm_cell_2/TanhTanh"lstm_2/while/lstm_cell_2/add_4:z:0*
T0*(
_output_shapes
:???????????
lstm_2/while/lstm_cell_2/mul_3Mul*lstm_2/while/lstm_cell_2/clip_by_value:z:0!lstm_2/while/lstm_cell_2/Tanh:y:0*
T0*(
_output_shapes
:???????????
lstm_2/while/lstm_cell_2/add_5AddV2"lstm_2/while/lstm_cell_2/mul_2:z:0"lstm_2/while/lstm_cell_2/mul_3:z:0*
T0*(
_output_shapes
:???????????
)lstm_2/while/lstm_cell_2/ReadVariableOp_3ReadVariableOp2lstm_2_while_lstm_cell_2_readvariableop_resource_0* 
_output_shapes
:
??*
dtype0
.lstm_2/while/lstm_cell_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       ?
0lstm_2/while/lstm_cell_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        ?
0lstm_2/while/lstm_cell_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
(lstm_2/while/lstm_cell_2/strided_slice_3StridedSlice1lstm_2/while/lstm_cell_2/ReadVariableOp_3:value:07lstm_2/while/lstm_cell_2/strided_slice_3/stack:output:09lstm_2/while/lstm_cell_2/strided_slice_3/stack_1:output:09lstm_2/while/lstm_cell_2/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
!lstm_2/while/lstm_cell_2/MatMul_7MatMullstm_2_while_placeholder_21lstm_2/while/lstm_cell_2/strided_slice_3:output:0*
T0*(
_output_shapes
:???????????
lstm_2/while/lstm_cell_2/add_6AddV2+lstm_2/while/lstm_cell_2/BiasAdd_3:output:0+lstm_2/while/lstm_cell_2/MatMul_7:product:0*
T0*(
_output_shapes
:??????????e
 lstm_2/while/lstm_cell_2/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>e
 lstm_2/while/lstm_cell_2/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ??
lstm_2/while/lstm_cell_2/Mul_4Mul"lstm_2/while/lstm_cell_2/add_6:z:0)lstm_2/while/lstm_cell_2/Const_4:output:0*
T0*(
_output_shapes
:???????????
lstm_2/while/lstm_cell_2/Add_7AddV2"lstm_2/while/lstm_cell_2/Mul_4:z:0)lstm_2/while/lstm_cell_2/Const_5:output:0*
T0*(
_output_shapes
:??????????w
2lstm_2/while/lstm_cell_2/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
0lstm_2/while/lstm_cell_2/clip_by_value_2/MinimumMinimum"lstm_2/while/lstm_cell_2/Add_7:z:0;lstm_2/while/lstm_cell_2/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:??????????o
*lstm_2/while/lstm_cell_2/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
(lstm_2/while/lstm_cell_2/clip_by_value_2Maximum4lstm_2/while/lstm_cell_2/clip_by_value_2/Minimum:z:03lstm_2/while/lstm_cell_2/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:??????????~
lstm_2/while/lstm_cell_2/Tanh_1Tanh"lstm_2/while/lstm_cell_2/add_5:z:0*
T0*(
_output_shapes
:???????????
lstm_2/while/lstm_cell_2/mul_5Mul,lstm_2/while/lstm_cell_2/clip_by_value_2:z:0#lstm_2/while/lstm_cell_2/Tanh_1:y:0*
T0*(
_output_shapes
:???????????
1lstm_2/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_2_while_placeholder_1lstm_2_while_placeholder"lstm_2/while/lstm_cell_2/mul_5:z:0*
_output_shapes
: *
element_dtype0:???T
lstm_2/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :q
lstm_2/while/addAddV2lstm_2_while_placeholderlstm_2/while/add/y:output:0*
T0*
_output_shapes
: V
lstm_2/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :?
lstm_2/while/add_1AddV2&lstm_2_while_lstm_2_while_loop_counterlstm_2/while/add_1/y:output:0*
T0*
_output_shapes
: n
lstm_2/while/IdentityIdentitylstm_2/while/add_1:z:0^lstm_2/while/NoOp*
T0*
_output_shapes
: ?
lstm_2/while/Identity_1Identity,lstm_2_while_lstm_2_while_maximum_iterations^lstm_2/while/NoOp*
T0*
_output_shapes
: n
lstm_2/while/Identity_2Identitylstm_2/while/add:z:0^lstm_2/while/NoOp*
T0*
_output_shapes
: ?
lstm_2/while/Identity_3IdentityAlstm_2/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_2/while/NoOp*
T0*
_output_shapes
: ?
lstm_2/while/Identity_4Identity"lstm_2/while/lstm_cell_2/mul_5:z:0^lstm_2/while/NoOp*
T0*(
_output_shapes
:???????????
lstm_2/while/Identity_5Identity"lstm_2/while/lstm_cell_2/add_5:z:0^lstm_2/while/NoOp*
T0*(
_output_shapes
:???????????
lstm_2/while/NoOpNoOp(^lstm_2/while/lstm_cell_2/ReadVariableOp*^lstm_2/while/lstm_cell_2/ReadVariableOp_1*^lstm_2/while/lstm_cell_2/ReadVariableOp_2*^lstm_2/while/lstm_cell_2/ReadVariableOp_3.^lstm_2/while/lstm_cell_2/split/ReadVariableOp0^lstm_2/while/lstm_cell_2/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "7
lstm_2_while_identitylstm_2/while/Identity:output:0";
lstm_2_while_identity_1 lstm_2/while/Identity_1:output:0";
lstm_2_while_identity_2 lstm_2/while/Identity_2:output:0";
lstm_2_while_identity_3 lstm_2/while/Identity_3:output:0";
lstm_2_while_identity_4 lstm_2/while/Identity_4:output:0";
lstm_2_while_identity_5 lstm_2/while/Identity_5:output:0"L
#lstm_2_while_lstm_2_strided_slice_1%lstm_2_while_lstm_2_strided_slice_1_0"f
0lstm_2_while_lstm_cell_2_readvariableop_resource2lstm_2_while_lstm_cell_2_readvariableop_resource_0"v
8lstm_2_while_lstm_cell_2_split_1_readvariableop_resource:lstm_2_while_lstm_cell_2_split_1_readvariableop_resource_0"r
6lstm_2_while_lstm_cell_2_split_readvariableop_resource8lstm_2_while_lstm_cell_2_split_readvariableop_resource_0"?
_lstm_2_while_tensorarrayv2read_tensorlistgetitem_lstm_2_tensorarrayunstack_tensorlistfromtensoralstm_2_while_tensorarrayv2read_tensorlistgetitem_lstm_2_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2R
'lstm_2/while/lstm_cell_2/ReadVariableOp'lstm_2/while/lstm_cell_2/ReadVariableOp2V
)lstm_2/while/lstm_cell_2/ReadVariableOp_1)lstm_2/while/lstm_cell_2/ReadVariableOp_12V
)lstm_2/while/lstm_cell_2/ReadVariableOp_2)lstm_2/while/lstm_cell_2/ReadVariableOp_22V
)lstm_2/while/lstm_cell_2/ReadVariableOp_3)lstm_2/while/lstm_cell_2/ReadVariableOp_32^
-lstm_2/while/lstm_cell_2/split/ReadVariableOp-lstm_2/while/lstm_cell_2/split/ReadVariableOp2b
/lstm_2/while/lstm_cell_2/split_1/ReadVariableOp/lstm_2/while/lstm_cell_2/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?{
?	
while_body_59017
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
1while_lstm_cell_3_split_readvariableop_resource_0:
??B
3while_lstm_cell_3_split_1_readvariableop_resource_0:	??
+while_lstm_cell_3_readvariableop_resource_0:
??
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
/while_lstm_cell_3_split_readvariableop_resource:
??@
1while_lstm_cell_3_split_1_readvariableop_resource:	?=
)while_lstm_cell_3_readvariableop_resource:
???? while/lstm_cell_3/ReadVariableOp?"while/lstm_cell_3/ReadVariableOp_1?"while/lstm_cell_3/ReadVariableOp_2?"while/lstm_cell_3/ReadVariableOp_3?&while/lstm_cell_3/split/ReadVariableOp?(while/lstm_cell_3/split_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype0c
!while/lstm_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
&while/lstm_cell_3/split/ReadVariableOpReadVariableOp1while_lstm_cell_3_split_readvariableop_resource_0* 
_output_shapes
:
??*
dtype0?
while/lstm_cell_3/splitSplit*while/lstm_cell_3/split/split_dim:output:0.while/lstm_cell_3/split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
??:
??:
??:
??*
	num_split?
while/lstm_cell_3/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_3/split:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_3/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_3/split:output:1*
T0*(
_output_shapes
:???????????
while/lstm_cell_3/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_3/split:output:2*
T0*(
_output_shapes
:???????????
while/lstm_cell_3/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_3/split:output:3*
T0*(
_output_shapes
:??????????e
#while/lstm_cell_3/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
(while/lstm_cell_3/split_1/ReadVariableOpReadVariableOp3while_lstm_cell_3_split_1_readvariableop_resource_0*
_output_shapes	
:?*
dtype0?
while/lstm_cell_3/split_1Split,while/lstm_cell_3/split_1/split_dim:output:00while/lstm_cell_3/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split?
while/lstm_cell_3/BiasAddBiasAdd"while/lstm_cell_3/MatMul:product:0"while/lstm_cell_3/split_1:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_3/BiasAdd_1BiasAdd$while/lstm_cell_3/MatMul_1:product:0"while/lstm_cell_3/split_1:output:1*
T0*(
_output_shapes
:???????????
while/lstm_cell_3/BiasAdd_2BiasAdd$while/lstm_cell_3/MatMul_2:product:0"while/lstm_cell_3/split_1:output:2*
T0*(
_output_shapes
:???????????
while/lstm_cell_3/BiasAdd_3BiasAdd$while/lstm_cell_3/MatMul_3:product:0"while/lstm_cell_3/split_1:output:3*
T0*(
_output_shapes
:???????????
 while/lstm_cell_3/ReadVariableOpReadVariableOp+while_lstm_cell_3_readvariableop_resource_0* 
_output_shapes
:
??*
dtype0v
%while/lstm_cell_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        x
'while/lstm_cell_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   x
'while/lstm_cell_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
while/lstm_cell_3/strided_sliceStridedSlice(while/lstm_cell_3/ReadVariableOp:value:0.while/lstm_cell_3/strided_slice/stack:output:00while/lstm_cell_3/strided_slice/stack_1:output:00while/lstm_cell_3/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
while/lstm_cell_3/MatMul_4MatMulwhile_placeholder_2(while/lstm_cell_3/strided_slice:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_3/addAddV2"while/lstm_cell_3/BiasAdd:output:0$while/lstm_cell_3/MatMul_4:product:0*
T0*(
_output_shapes
:??????????\
while/lstm_cell_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>^
while/lstm_cell_3/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ??
while/lstm_cell_3/MulMulwhile/lstm_cell_3/add:z:0 while/lstm_cell_3/Const:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_3/Add_1AddV2while/lstm_cell_3/Mul:z:0"while/lstm_cell_3/Const_1:output:0*
T0*(
_output_shapes
:??????????n
)while/lstm_cell_3/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
'while/lstm_cell_3/clip_by_value/MinimumMinimumwhile/lstm_cell_3/Add_1:z:02while/lstm_cell_3/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:??????????f
!while/lstm_cell_3/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
while/lstm_cell_3/clip_by_valueMaximum+while/lstm_cell_3/clip_by_value/Minimum:z:0*while/lstm_cell_3/clip_by_value/y:output:0*
T0*(
_output_shapes
:???????????
"while/lstm_cell_3/ReadVariableOp_1ReadVariableOp+while_lstm_cell_3_readvariableop_resource_0* 
_output_shapes
:
??*
dtype0x
'while/lstm_cell_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   z
)while/lstm_cell_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       z
)while/lstm_cell_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
!while/lstm_cell_3/strided_slice_1StridedSlice*while/lstm_cell_3/ReadVariableOp_1:value:00while/lstm_cell_3/strided_slice_1/stack:output:02while/lstm_cell_3/strided_slice_1/stack_1:output:02while/lstm_cell_3/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
while/lstm_cell_3/MatMul_5MatMulwhile_placeholder_2*while/lstm_cell_3/strided_slice_1:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_3/add_2AddV2$while/lstm_cell_3/BiasAdd_1:output:0$while/lstm_cell_3/MatMul_5:product:0*
T0*(
_output_shapes
:??????????^
while/lstm_cell_3/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>^
while/lstm_cell_3/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ??
while/lstm_cell_3/Mul_1Mulwhile/lstm_cell_3/add_2:z:0"while/lstm_cell_3/Const_2:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_3/Add_3AddV2while/lstm_cell_3/Mul_1:z:0"while/lstm_cell_3/Const_3:output:0*
T0*(
_output_shapes
:??????????p
+while/lstm_cell_3/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
)while/lstm_cell_3/clip_by_value_1/MinimumMinimumwhile/lstm_cell_3/Add_3:z:04while/lstm_cell_3/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:??????????h
#while/lstm_cell_3/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
!while/lstm_cell_3/clip_by_value_1Maximum-while/lstm_cell_3/clip_by_value_1/Minimum:z:0,while/lstm_cell_3/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_3/mul_2Mul%while/lstm_cell_3/clip_by_value_1:z:0while_placeholder_3*
T0*(
_output_shapes
:???????????
"while/lstm_cell_3/ReadVariableOp_2ReadVariableOp+while_lstm_cell_3_readvariableop_resource_0* 
_output_shapes
:
??*
dtype0x
'while/lstm_cell_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       z
)while/lstm_cell_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  z
)while/lstm_cell_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
!while/lstm_cell_3/strided_slice_2StridedSlice*while/lstm_cell_3/ReadVariableOp_2:value:00while/lstm_cell_3/strided_slice_2/stack:output:02while/lstm_cell_3/strided_slice_2/stack_1:output:02while/lstm_cell_3/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
while/lstm_cell_3/MatMul_6MatMulwhile_placeholder_2*while/lstm_cell_3/strided_slice_2:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_3/add_4AddV2$while/lstm_cell_3/BiasAdd_2:output:0$while/lstm_cell_3/MatMul_6:product:0*
T0*(
_output_shapes
:??????????n
while/lstm_cell_3/TanhTanhwhile/lstm_cell_3/add_4:z:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_3/mul_3Mul#while/lstm_cell_3/clip_by_value:z:0while/lstm_cell_3/Tanh:y:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_3/add_5AddV2while/lstm_cell_3/mul_2:z:0while/lstm_cell_3/mul_3:z:0*
T0*(
_output_shapes
:???????????
"while/lstm_cell_3/ReadVariableOp_3ReadVariableOp+while_lstm_cell_3_readvariableop_resource_0* 
_output_shapes
:
??*
dtype0x
'while/lstm_cell_3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  z
)while/lstm_cell_3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        z
)while/lstm_cell_3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
!while/lstm_cell_3/strided_slice_3StridedSlice*while/lstm_cell_3/ReadVariableOp_3:value:00while/lstm_cell_3/strided_slice_3/stack:output:02while/lstm_cell_3/strided_slice_3/stack_1:output:02while/lstm_cell_3/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
while/lstm_cell_3/MatMul_7MatMulwhile_placeholder_2*while/lstm_cell_3/strided_slice_3:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_3/add_6AddV2$while/lstm_cell_3/BiasAdd_3:output:0$while/lstm_cell_3/MatMul_7:product:0*
T0*(
_output_shapes
:??????????^
while/lstm_cell_3/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>^
while/lstm_cell_3/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ??
while/lstm_cell_3/Mul_4Mulwhile/lstm_cell_3/add_6:z:0"while/lstm_cell_3/Const_4:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_3/Add_7AddV2while/lstm_cell_3/Mul_4:z:0"while/lstm_cell_3/Const_5:output:0*
T0*(
_output_shapes
:??????????p
+while/lstm_cell_3/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
)while/lstm_cell_3/clip_by_value_2/MinimumMinimumwhile/lstm_cell_3/Add_7:z:04while/lstm_cell_3/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:??????????h
#while/lstm_cell_3/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
!while/lstm_cell_3/clip_by_value_2Maximum-while/lstm_cell_3/clip_by_value_2/Minimum:z:0,while/lstm_cell_3/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:??????????p
while/lstm_cell_3/Tanh_1Tanhwhile/lstm_cell_3/add_5:z:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_3/mul_5Mul%while/lstm_cell_3/clip_by_value_2:z:0while/lstm_cell_3/Tanh_1:y:0*
T0*(
_output_shapes
:???????????
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_3/mul_5:z:0*
_output_shapes
: *
element_dtype0:???M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: y
while/Identity_4Identitywhile/lstm_cell_3/mul_5:z:0^while/NoOp*
T0*(
_output_shapes
:??????????y
while/Identity_5Identitywhile/lstm_cell_3/add_5:z:0^while/NoOp*
T0*(
_output_shapes
:???????????

while/NoOpNoOp!^while/lstm_cell_3/ReadVariableOp#^while/lstm_cell_3/ReadVariableOp_1#^while/lstm_cell_3/ReadVariableOp_2#^while/lstm_cell_3/ReadVariableOp_3'^while/lstm_cell_3/split/ReadVariableOp)^while/lstm_cell_3/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"X
)while_lstm_cell_3_readvariableop_resource+while_lstm_cell_3_readvariableop_resource_0"h
1while_lstm_cell_3_split_1_readvariableop_resource3while_lstm_cell_3_split_1_readvariableop_resource_0"d
/while_lstm_cell_3_split_readvariableop_resource1while_lstm_cell_3_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2D
 while/lstm_cell_3/ReadVariableOp while/lstm_cell_3/ReadVariableOp2H
"while/lstm_cell_3/ReadVariableOp_1"while/lstm_cell_3/ReadVariableOp_12H
"while/lstm_cell_3/ReadVariableOp_2"while/lstm_cell_3/ReadVariableOp_22H
"while/lstm_cell_3/ReadVariableOp_3"while/lstm_cell_3/ReadVariableOp_32P
&while/lstm_cell_3/split/ReadVariableOp&while/lstm_cell_3/split/ReadVariableOp2T
(while/lstm_cell_3/split_1/ReadVariableOp(while/lstm_cell_3/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?

?
B__inference_dense_3_layer_call_and_return_conditional_losses_55429

inputs1
matmul_readvariableop_resource:	?-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
while_cond_57436
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_57436___redundant_placeholder03
/while_while_cond_57436___redundant_placeholder13
/while_while_cond_57436___redundant_placeholder23
/while_while_cond_57436___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?
?
while_cond_59272
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_59272___redundant_placeholder03
/while_while_cond_59272___redundant_placeholder13
/while_while_cond_59272___redundant_placeholder23
/while_while_cond_59272___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?{
?	
while_body_58505
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
1while_lstm_cell_3_split_readvariableop_resource_0:
??B
3while_lstm_cell_3_split_1_readvariableop_resource_0:	??
+while_lstm_cell_3_readvariableop_resource_0:
??
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
/while_lstm_cell_3_split_readvariableop_resource:
??@
1while_lstm_cell_3_split_1_readvariableop_resource:	?=
)while_lstm_cell_3_readvariableop_resource:
???? while/lstm_cell_3/ReadVariableOp?"while/lstm_cell_3/ReadVariableOp_1?"while/lstm_cell_3/ReadVariableOp_2?"while/lstm_cell_3/ReadVariableOp_3?&while/lstm_cell_3/split/ReadVariableOp?(while/lstm_cell_3/split_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype0c
!while/lstm_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
&while/lstm_cell_3/split/ReadVariableOpReadVariableOp1while_lstm_cell_3_split_readvariableop_resource_0* 
_output_shapes
:
??*
dtype0?
while/lstm_cell_3/splitSplit*while/lstm_cell_3/split/split_dim:output:0.while/lstm_cell_3/split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
??:
??:
??:
??*
	num_split?
while/lstm_cell_3/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_3/split:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_3/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_3/split:output:1*
T0*(
_output_shapes
:???????????
while/lstm_cell_3/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_3/split:output:2*
T0*(
_output_shapes
:???????????
while/lstm_cell_3/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_3/split:output:3*
T0*(
_output_shapes
:??????????e
#while/lstm_cell_3/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
(while/lstm_cell_3/split_1/ReadVariableOpReadVariableOp3while_lstm_cell_3_split_1_readvariableop_resource_0*
_output_shapes	
:?*
dtype0?
while/lstm_cell_3/split_1Split,while/lstm_cell_3/split_1/split_dim:output:00while/lstm_cell_3/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split?
while/lstm_cell_3/BiasAddBiasAdd"while/lstm_cell_3/MatMul:product:0"while/lstm_cell_3/split_1:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_3/BiasAdd_1BiasAdd$while/lstm_cell_3/MatMul_1:product:0"while/lstm_cell_3/split_1:output:1*
T0*(
_output_shapes
:???????????
while/lstm_cell_3/BiasAdd_2BiasAdd$while/lstm_cell_3/MatMul_2:product:0"while/lstm_cell_3/split_1:output:2*
T0*(
_output_shapes
:???????????
while/lstm_cell_3/BiasAdd_3BiasAdd$while/lstm_cell_3/MatMul_3:product:0"while/lstm_cell_3/split_1:output:3*
T0*(
_output_shapes
:???????????
 while/lstm_cell_3/ReadVariableOpReadVariableOp+while_lstm_cell_3_readvariableop_resource_0* 
_output_shapes
:
??*
dtype0v
%while/lstm_cell_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        x
'while/lstm_cell_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   x
'while/lstm_cell_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
while/lstm_cell_3/strided_sliceStridedSlice(while/lstm_cell_3/ReadVariableOp:value:0.while/lstm_cell_3/strided_slice/stack:output:00while/lstm_cell_3/strided_slice/stack_1:output:00while/lstm_cell_3/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
while/lstm_cell_3/MatMul_4MatMulwhile_placeholder_2(while/lstm_cell_3/strided_slice:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_3/addAddV2"while/lstm_cell_3/BiasAdd:output:0$while/lstm_cell_3/MatMul_4:product:0*
T0*(
_output_shapes
:??????????\
while/lstm_cell_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>^
while/lstm_cell_3/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ??
while/lstm_cell_3/MulMulwhile/lstm_cell_3/add:z:0 while/lstm_cell_3/Const:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_3/Add_1AddV2while/lstm_cell_3/Mul:z:0"while/lstm_cell_3/Const_1:output:0*
T0*(
_output_shapes
:??????????n
)while/lstm_cell_3/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
'while/lstm_cell_3/clip_by_value/MinimumMinimumwhile/lstm_cell_3/Add_1:z:02while/lstm_cell_3/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:??????????f
!while/lstm_cell_3/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
while/lstm_cell_3/clip_by_valueMaximum+while/lstm_cell_3/clip_by_value/Minimum:z:0*while/lstm_cell_3/clip_by_value/y:output:0*
T0*(
_output_shapes
:???????????
"while/lstm_cell_3/ReadVariableOp_1ReadVariableOp+while_lstm_cell_3_readvariableop_resource_0* 
_output_shapes
:
??*
dtype0x
'while/lstm_cell_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   z
)while/lstm_cell_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       z
)while/lstm_cell_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
!while/lstm_cell_3/strided_slice_1StridedSlice*while/lstm_cell_3/ReadVariableOp_1:value:00while/lstm_cell_3/strided_slice_1/stack:output:02while/lstm_cell_3/strided_slice_1/stack_1:output:02while/lstm_cell_3/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
while/lstm_cell_3/MatMul_5MatMulwhile_placeholder_2*while/lstm_cell_3/strided_slice_1:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_3/add_2AddV2$while/lstm_cell_3/BiasAdd_1:output:0$while/lstm_cell_3/MatMul_5:product:0*
T0*(
_output_shapes
:??????????^
while/lstm_cell_3/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>^
while/lstm_cell_3/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ??
while/lstm_cell_3/Mul_1Mulwhile/lstm_cell_3/add_2:z:0"while/lstm_cell_3/Const_2:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_3/Add_3AddV2while/lstm_cell_3/Mul_1:z:0"while/lstm_cell_3/Const_3:output:0*
T0*(
_output_shapes
:??????????p
+while/lstm_cell_3/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
)while/lstm_cell_3/clip_by_value_1/MinimumMinimumwhile/lstm_cell_3/Add_3:z:04while/lstm_cell_3/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:??????????h
#while/lstm_cell_3/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
!while/lstm_cell_3/clip_by_value_1Maximum-while/lstm_cell_3/clip_by_value_1/Minimum:z:0,while/lstm_cell_3/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_3/mul_2Mul%while/lstm_cell_3/clip_by_value_1:z:0while_placeholder_3*
T0*(
_output_shapes
:???????????
"while/lstm_cell_3/ReadVariableOp_2ReadVariableOp+while_lstm_cell_3_readvariableop_resource_0* 
_output_shapes
:
??*
dtype0x
'while/lstm_cell_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       z
)while/lstm_cell_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  z
)while/lstm_cell_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
!while/lstm_cell_3/strided_slice_2StridedSlice*while/lstm_cell_3/ReadVariableOp_2:value:00while/lstm_cell_3/strided_slice_2/stack:output:02while/lstm_cell_3/strided_slice_2/stack_1:output:02while/lstm_cell_3/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
while/lstm_cell_3/MatMul_6MatMulwhile_placeholder_2*while/lstm_cell_3/strided_slice_2:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_3/add_4AddV2$while/lstm_cell_3/BiasAdd_2:output:0$while/lstm_cell_3/MatMul_6:product:0*
T0*(
_output_shapes
:??????????n
while/lstm_cell_3/TanhTanhwhile/lstm_cell_3/add_4:z:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_3/mul_3Mul#while/lstm_cell_3/clip_by_value:z:0while/lstm_cell_3/Tanh:y:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_3/add_5AddV2while/lstm_cell_3/mul_2:z:0while/lstm_cell_3/mul_3:z:0*
T0*(
_output_shapes
:???????????
"while/lstm_cell_3/ReadVariableOp_3ReadVariableOp+while_lstm_cell_3_readvariableop_resource_0* 
_output_shapes
:
??*
dtype0x
'while/lstm_cell_3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  z
)while/lstm_cell_3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        z
)while/lstm_cell_3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
!while/lstm_cell_3/strided_slice_3StridedSlice*while/lstm_cell_3/ReadVariableOp_3:value:00while/lstm_cell_3/strided_slice_3/stack:output:02while/lstm_cell_3/strided_slice_3/stack_1:output:02while/lstm_cell_3/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
while/lstm_cell_3/MatMul_7MatMulwhile_placeholder_2*while/lstm_cell_3/strided_slice_3:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_3/add_6AddV2$while/lstm_cell_3/BiasAdd_3:output:0$while/lstm_cell_3/MatMul_7:product:0*
T0*(
_output_shapes
:??????????^
while/lstm_cell_3/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>^
while/lstm_cell_3/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ??
while/lstm_cell_3/Mul_4Mulwhile/lstm_cell_3/add_6:z:0"while/lstm_cell_3/Const_4:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_3/Add_7AddV2while/lstm_cell_3/Mul_4:z:0"while/lstm_cell_3/Const_5:output:0*
T0*(
_output_shapes
:??????????p
+while/lstm_cell_3/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
)while/lstm_cell_3/clip_by_value_2/MinimumMinimumwhile/lstm_cell_3/Add_7:z:04while/lstm_cell_3/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:??????????h
#while/lstm_cell_3/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
!while/lstm_cell_3/clip_by_value_2Maximum-while/lstm_cell_3/clip_by_value_2/Minimum:z:0,while/lstm_cell_3/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:??????????p
while/lstm_cell_3/Tanh_1Tanhwhile/lstm_cell_3/add_5:z:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_3/mul_5Mul%while/lstm_cell_3/clip_by_value_2:z:0while/lstm_cell_3/Tanh_1:y:0*
T0*(
_output_shapes
:???????????
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_3/mul_5:z:0*
_output_shapes
: *
element_dtype0:???M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: y
while/Identity_4Identitywhile/lstm_cell_3/mul_5:z:0^while/NoOp*
T0*(
_output_shapes
:??????????y
while/Identity_5Identitywhile/lstm_cell_3/add_5:z:0^while/NoOp*
T0*(
_output_shapes
:???????????

while/NoOpNoOp!^while/lstm_cell_3/ReadVariableOp#^while/lstm_cell_3/ReadVariableOp_1#^while/lstm_cell_3/ReadVariableOp_2#^while/lstm_cell_3/ReadVariableOp_3'^while/lstm_cell_3/split/ReadVariableOp)^while/lstm_cell_3/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"X
)while_lstm_cell_3_readvariableop_resource+while_lstm_cell_3_readvariableop_resource_0"h
1while_lstm_cell_3_split_1_readvariableop_resource3while_lstm_cell_3_split_1_readvariableop_resource_0"d
/while_lstm_cell_3_split_readvariableop_resource1while_lstm_cell_3_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2D
 while/lstm_cell_3/ReadVariableOp while/lstm_cell_3/ReadVariableOp2H
"while/lstm_cell_3/ReadVariableOp_1"while/lstm_cell_3/ReadVariableOp_12H
"while/lstm_cell_3/ReadVariableOp_2"while/lstm_cell_3/ReadVariableOp_22H
"while/lstm_cell_3/ReadVariableOp_3"while/lstm_cell_3/ReadVariableOp_32P
&while/lstm_cell_3/split/ReadVariableOp&while/lstm_cell_3/split/ReadVariableOp2T
(while/lstm_cell_3/split_1/ReadVariableOp(while/lstm_cell_3/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?
?
while_cond_54081
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_54081___redundant_placeholder03
/while_while_cond_54081___redundant_placeholder13
/while_while_cond_54081___redundant_placeholder23
/while_while_cond_54081___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?
?
while_cond_58760
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_58760___redundant_placeholder03
/while_while_cond_58760___redundant_placeholder13
/while_while_cond_58760___redundant_placeholder23
/while_while_cond_58760___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?
?
G__inference_sequential_1_layer_call_and_return_conditional_losses_56194
lstm_2_input
lstm_2_56169:	?
lstm_2_56171:	? 
lstm_2_56173:
?? 
lstm_3_56176:
??
lstm_3_56178:	? 
lstm_3_56180:
??!
dense_2_56183:
??
dense_2_56185:	? 
dense_3_56188:	?
dense_3_56190:
identity??dense_2/StatefulPartitionedCall?dense_3/StatefulPartitionedCall?lstm_2/StatefulPartitionedCall?lstm_3/StatefulPartitionedCall?
lstm_2/StatefulPartitionedCallStatefulPartitionedCalllstm_2_inputlstm_2_56169lstm_2_56171lstm_2_56173*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_lstm_2_layer_call_and_return_conditional_losses_56026?
lstm_3/StatefulPartitionedCallStatefulPartitionedCall'lstm_2/StatefulPartitionedCall:output:0lstm_3_56176lstm_3_56178lstm_3_56180*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_lstm_3_layer_call_and_return_conditional_losses_55748?
dense_2/StatefulPartitionedCallStatefulPartitionedCall'lstm_3/StatefulPartitionedCall:output:0dense_2_56183dense_2_56185*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_55412?
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_56188dense_3_56190*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_55429w
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall^lstm_2/StatefulPartitionedCall^lstm_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????: : : : : : : : : : 2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2@
lstm_2/StatefulPartitionedCalllstm_2/StatefulPartitionedCall2@
lstm_3/StatefulPartitionedCalllstm_3/StatefulPartitionedCall:Y U
+
_output_shapes
:?????????
&
_user_specified_namelstm_2_input
?7
?
A__inference_lstm_3_layer_call_and_return_conditional_losses_54612

inputs%
lstm_cell_3_54531:
?? 
lstm_cell_3_54533:	?%
lstm_cell_3_54535:
??
identity??#lstm_cell_3/StatefulPartitionedCall?while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????S
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          w
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:???????????????????D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask?
#lstm_cell_3/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_3_54531lstm_cell_3_54533lstm_cell_3_54535*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:??????????:??????????:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_lstm_cell_3_layer_call_and_return_conditional_losses_54530n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   ?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_3_54531lstm_cell_3_54533lstm_cell_3_54535*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_54544*
condR
while_cond_54543*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:???????????????????*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:???????????????????h
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*(
_output_shapes
:??????????t
NoOpNoOp$^lstm_cell_3/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':???????????????????: : : 2J
#lstm_cell_3/StatefulPartitionedCall#lstm_cell_3/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs
?
?
&__inference_lstm_3_layer_call_fn_59435
inputs_0
unknown:
??
	unknown_0:	?
	unknown_1:
??
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_lstm_3_layer_call_and_return_conditional_losses_54859p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':???????????????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
5
_output_shapes#
!:???????????????????
"
_user_specified_name
inputs/0
?"
?
while_body_54329
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0,
while_lstm_cell_2_54353_0:	?(
while_lstm_cell_2_54355_0:	?-
while_lstm_cell_2_54357_0:
??
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor*
while_lstm_cell_2_54353:	?&
while_lstm_cell_2_54355:	?+
while_lstm_cell_2_54357:
????)while/lstm_cell_2/StatefulPartitionedCall?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0?
)while/lstm_cell_2/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_2_54353_0while_lstm_cell_2_54355_0while_lstm_cell_2_54357_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:??????????:??????????:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_lstm_cell_2_layer_call_and_return_conditional_losses_54270?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder2while/lstm_cell_2/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:???M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_4Identity2while/lstm_cell_2/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:???????????
while/Identity_5Identity2while/lstm_cell_2/StatefulPartitionedCall:output:2^while/NoOp*
T0*(
_output_shapes
:??????????x

while/NoOpNoOp*^while/lstm_cell_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"4
while_lstm_cell_2_54353while_lstm_cell_2_54353_0"4
while_lstm_cell_2_54355while_lstm_cell_2_54355_0"4
while_lstm_cell_2_54357while_lstm_cell_2_54357_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2V
)while/lstm_cell_2/StatefulPartitionedCall)while/lstm_cell_2/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?{
?	
while_body_57693
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_lstm_cell_2_split_readvariableop_resource_0:	?B
3while_lstm_cell_2_split_1_readvariableop_resource_0:	??
+while_lstm_cell_2_readvariableop_resource_0:
??
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_lstm_cell_2_split_readvariableop_resource:	?@
1while_lstm_cell_2_split_1_readvariableop_resource:	?=
)while_lstm_cell_2_readvariableop_resource:
???? while/lstm_cell_2/ReadVariableOp?"while/lstm_cell_2/ReadVariableOp_1?"while/lstm_cell_2/ReadVariableOp_2?"while/lstm_cell_2/ReadVariableOp_3?&while/lstm_cell_2/split/ReadVariableOp?(while/lstm_cell_2/split_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0c
!while/lstm_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
&while/lstm_cell_2/split/ReadVariableOpReadVariableOp1while_lstm_cell_2_split_readvariableop_resource_0*
_output_shapes
:	?*
dtype0?
while/lstm_cell_2/splitSplit*while/lstm_cell_2/split/split_dim:output:0.while/lstm_cell_2/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	?:	?:	?:	?*
	num_split?
while/lstm_cell_2/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_2/split:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_2/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_2/split:output:1*
T0*(
_output_shapes
:???????????
while/lstm_cell_2/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_2/split:output:2*
T0*(
_output_shapes
:???????????
while/lstm_cell_2/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_2/split:output:3*
T0*(
_output_shapes
:??????????e
#while/lstm_cell_2/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
(while/lstm_cell_2/split_1/ReadVariableOpReadVariableOp3while_lstm_cell_2_split_1_readvariableop_resource_0*
_output_shapes	
:?*
dtype0?
while/lstm_cell_2/split_1Split,while/lstm_cell_2/split_1/split_dim:output:00while/lstm_cell_2/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split?
while/lstm_cell_2/BiasAddBiasAdd"while/lstm_cell_2/MatMul:product:0"while/lstm_cell_2/split_1:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_2/BiasAdd_1BiasAdd$while/lstm_cell_2/MatMul_1:product:0"while/lstm_cell_2/split_1:output:1*
T0*(
_output_shapes
:???????????
while/lstm_cell_2/BiasAdd_2BiasAdd$while/lstm_cell_2/MatMul_2:product:0"while/lstm_cell_2/split_1:output:2*
T0*(
_output_shapes
:???????????
while/lstm_cell_2/BiasAdd_3BiasAdd$while/lstm_cell_2/MatMul_3:product:0"while/lstm_cell_2/split_1:output:3*
T0*(
_output_shapes
:???????????
 while/lstm_cell_2/ReadVariableOpReadVariableOp+while_lstm_cell_2_readvariableop_resource_0* 
_output_shapes
:
??*
dtype0v
%while/lstm_cell_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        x
'while/lstm_cell_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       x
'while/lstm_cell_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
while/lstm_cell_2/strided_sliceStridedSlice(while/lstm_cell_2/ReadVariableOp:value:0.while/lstm_cell_2/strided_slice/stack:output:00while/lstm_cell_2/strided_slice/stack_1:output:00while/lstm_cell_2/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
while/lstm_cell_2/MatMul_4MatMulwhile_placeholder_2(while/lstm_cell_2/strided_slice:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_2/addAddV2"while/lstm_cell_2/BiasAdd:output:0$while/lstm_cell_2/MatMul_4:product:0*
T0*(
_output_shapes
:??????????\
while/lstm_cell_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>^
while/lstm_cell_2/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ??
while/lstm_cell_2/MulMulwhile/lstm_cell_2/add:z:0 while/lstm_cell_2/Const:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_2/Add_1AddV2while/lstm_cell_2/Mul:z:0"while/lstm_cell_2/Const_1:output:0*
T0*(
_output_shapes
:??????????n
)while/lstm_cell_2/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
'while/lstm_cell_2/clip_by_value/MinimumMinimumwhile/lstm_cell_2/Add_1:z:02while/lstm_cell_2/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:??????????f
!while/lstm_cell_2/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
while/lstm_cell_2/clip_by_valueMaximum+while/lstm_cell_2/clip_by_value/Minimum:z:0*while/lstm_cell_2/clip_by_value/y:output:0*
T0*(
_output_shapes
:???????????
"while/lstm_cell_2/ReadVariableOp_1ReadVariableOp+while_lstm_cell_2_readvariableop_resource_0* 
_output_shapes
:
??*
dtype0x
'while/lstm_cell_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       z
)while/lstm_cell_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       z
)while/lstm_cell_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
!while/lstm_cell_2/strided_slice_1StridedSlice*while/lstm_cell_2/ReadVariableOp_1:value:00while/lstm_cell_2/strided_slice_1/stack:output:02while/lstm_cell_2/strided_slice_1/stack_1:output:02while/lstm_cell_2/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
while/lstm_cell_2/MatMul_5MatMulwhile_placeholder_2*while/lstm_cell_2/strided_slice_1:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_2/add_2AddV2$while/lstm_cell_2/BiasAdd_1:output:0$while/lstm_cell_2/MatMul_5:product:0*
T0*(
_output_shapes
:??????????^
while/lstm_cell_2/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>^
while/lstm_cell_2/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ??
while/lstm_cell_2/Mul_1Mulwhile/lstm_cell_2/add_2:z:0"while/lstm_cell_2/Const_2:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_2/Add_3AddV2while/lstm_cell_2/Mul_1:z:0"while/lstm_cell_2/Const_3:output:0*
T0*(
_output_shapes
:??????????p
+while/lstm_cell_2/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
)while/lstm_cell_2/clip_by_value_1/MinimumMinimumwhile/lstm_cell_2/Add_3:z:04while/lstm_cell_2/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:??????????h
#while/lstm_cell_2/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
!while/lstm_cell_2/clip_by_value_1Maximum-while/lstm_cell_2/clip_by_value_1/Minimum:z:0,while/lstm_cell_2/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_2/mul_2Mul%while/lstm_cell_2/clip_by_value_1:z:0while_placeholder_3*
T0*(
_output_shapes
:???????????
"while/lstm_cell_2/ReadVariableOp_2ReadVariableOp+while_lstm_cell_2_readvariableop_resource_0* 
_output_shapes
:
??*
dtype0x
'while/lstm_cell_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       z
)while/lstm_cell_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       z
)while/lstm_cell_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
!while/lstm_cell_2/strided_slice_2StridedSlice*while/lstm_cell_2/ReadVariableOp_2:value:00while/lstm_cell_2/strided_slice_2/stack:output:02while/lstm_cell_2/strided_slice_2/stack_1:output:02while/lstm_cell_2/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
while/lstm_cell_2/MatMul_6MatMulwhile_placeholder_2*while/lstm_cell_2/strided_slice_2:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_2/add_4AddV2$while/lstm_cell_2/BiasAdd_2:output:0$while/lstm_cell_2/MatMul_6:product:0*
T0*(
_output_shapes
:??????????n
while/lstm_cell_2/TanhTanhwhile/lstm_cell_2/add_4:z:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_2/mul_3Mul#while/lstm_cell_2/clip_by_value:z:0while/lstm_cell_2/Tanh:y:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_2/add_5AddV2while/lstm_cell_2/mul_2:z:0while/lstm_cell_2/mul_3:z:0*
T0*(
_output_shapes
:???????????
"while/lstm_cell_2/ReadVariableOp_3ReadVariableOp+while_lstm_cell_2_readvariableop_resource_0* 
_output_shapes
:
??*
dtype0x
'while/lstm_cell_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       z
)while/lstm_cell_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        z
)while/lstm_cell_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
!while/lstm_cell_2/strided_slice_3StridedSlice*while/lstm_cell_2/ReadVariableOp_3:value:00while/lstm_cell_2/strided_slice_3/stack:output:02while/lstm_cell_2/strided_slice_3/stack_1:output:02while/lstm_cell_2/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
while/lstm_cell_2/MatMul_7MatMulwhile_placeholder_2*while/lstm_cell_2/strided_slice_3:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_2/add_6AddV2$while/lstm_cell_2/BiasAdd_3:output:0$while/lstm_cell_2/MatMul_7:product:0*
T0*(
_output_shapes
:??????????^
while/lstm_cell_2/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>^
while/lstm_cell_2/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ??
while/lstm_cell_2/Mul_4Mulwhile/lstm_cell_2/add_6:z:0"while/lstm_cell_2/Const_4:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_2/Add_7AddV2while/lstm_cell_2/Mul_4:z:0"while/lstm_cell_2/Const_5:output:0*
T0*(
_output_shapes
:??????????p
+while/lstm_cell_2/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
)while/lstm_cell_2/clip_by_value_2/MinimumMinimumwhile/lstm_cell_2/Add_7:z:04while/lstm_cell_2/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:??????????h
#while/lstm_cell_2/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
!while/lstm_cell_2/clip_by_value_2Maximum-while/lstm_cell_2/clip_by_value_2/Minimum:z:0,while/lstm_cell_2/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:??????????p
while/lstm_cell_2/Tanh_1Tanhwhile/lstm_cell_2/add_5:z:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_2/mul_5Mul%while/lstm_cell_2/clip_by_value_2:z:0while/lstm_cell_2/Tanh_1:y:0*
T0*(
_output_shapes
:???????????
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_2/mul_5:z:0*
_output_shapes
: *
element_dtype0:???M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: y
while/Identity_4Identitywhile/lstm_cell_2/mul_5:z:0^while/NoOp*
T0*(
_output_shapes
:??????????y
while/Identity_5Identitywhile/lstm_cell_2/add_5:z:0^while/NoOp*
T0*(
_output_shapes
:???????????

while/NoOpNoOp!^while/lstm_cell_2/ReadVariableOp#^while/lstm_cell_2/ReadVariableOp_1#^while/lstm_cell_2/ReadVariableOp_2#^while/lstm_cell_2/ReadVariableOp_3'^while/lstm_cell_2/split/ReadVariableOp)^while/lstm_cell_2/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"X
)while_lstm_cell_2_readvariableop_resource+while_lstm_cell_2_readvariableop_resource_0"h
1while_lstm_cell_2_split_1_readvariableop_resource3while_lstm_cell_2_split_1_readvariableop_resource_0"d
/while_lstm_cell_2_split_readvariableop_resource1while_lstm_cell_2_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2D
 while/lstm_cell_2/ReadVariableOp while/lstm_cell_2/ReadVariableOp2H
"while/lstm_cell_2/ReadVariableOp_1"while/lstm_cell_2/ReadVariableOp_12H
"while/lstm_cell_2/ReadVariableOp_2"while/lstm_cell_2/ReadVariableOp_22H
"while/lstm_cell_2/ReadVariableOp_3"while/lstm_cell_2/ReadVariableOp_32P
&while/lstm_cell_2/split/ReadVariableOp&while/lstm_cell_2/split/ReadVariableOp2T
(while/lstm_cell_2/split_1/ReadVariableOp(while/lstm_cell_2/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?
?
+__inference_lstm_cell_2_layer_call_fn_59709

inputs
states_0
states_1
unknown:	?
	unknown_0:	?
	unknown_1:
??
identity

identity_1

identity_2??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:??????????:??????????:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_lstm_cell_2_layer_call_and_return_conditional_losses_54270p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????r

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:??????????r

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:?????????:??????????:??????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/0:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/1
?	
?
lstm_3_while_cond_57116*
&lstm_3_while_lstm_3_while_loop_counter0
,lstm_3_while_lstm_3_while_maximum_iterations
lstm_3_while_placeholder
lstm_3_while_placeholder_1
lstm_3_while_placeholder_2
lstm_3_while_placeholder_3,
(lstm_3_while_less_lstm_3_strided_slice_1A
=lstm_3_while_lstm_3_while_cond_57116___redundant_placeholder0A
=lstm_3_while_lstm_3_while_cond_57116___redundant_placeholder1A
=lstm_3_while_lstm_3_while_cond_57116___redundant_placeholder2A
=lstm_3_while_lstm_3_while_cond_57116___redundant_placeholder3
lstm_3_while_identity
~
lstm_3/while/LessLesslstm_3_while_placeholder(lstm_3_while_less_lstm_3_strided_slice_1*
T0*
_output_shapes
: Y
lstm_3/while/IdentityIdentitylstm_3/while/Less:z:0*
T0
*
_output_shapes
: "7
lstm_3_while_identitylstm_3/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?

?
,__inference_sequential_1_layer_call_fn_57321

inputs
unknown:	?
	unknown_0:	?
	unknown_1:
??
	unknown_2:
??
	unknown_3:	?
	unknown_4:
??
	unknown_5:
??
	unknown_6:	?
	unknown_7:	?
	unknown_8:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_56090o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?{
?	
while_body_55886
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_lstm_cell_2_split_readvariableop_resource_0:	?B
3while_lstm_cell_2_split_1_readvariableop_resource_0:	??
+while_lstm_cell_2_readvariableop_resource_0:
??
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_lstm_cell_2_split_readvariableop_resource:	?@
1while_lstm_cell_2_split_1_readvariableop_resource:	?=
)while_lstm_cell_2_readvariableop_resource:
???? while/lstm_cell_2/ReadVariableOp?"while/lstm_cell_2/ReadVariableOp_1?"while/lstm_cell_2/ReadVariableOp_2?"while/lstm_cell_2/ReadVariableOp_3?&while/lstm_cell_2/split/ReadVariableOp?(while/lstm_cell_2/split_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0c
!while/lstm_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
&while/lstm_cell_2/split/ReadVariableOpReadVariableOp1while_lstm_cell_2_split_readvariableop_resource_0*
_output_shapes
:	?*
dtype0?
while/lstm_cell_2/splitSplit*while/lstm_cell_2/split/split_dim:output:0.while/lstm_cell_2/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	?:	?:	?:	?*
	num_split?
while/lstm_cell_2/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_2/split:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_2/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_2/split:output:1*
T0*(
_output_shapes
:???????????
while/lstm_cell_2/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_2/split:output:2*
T0*(
_output_shapes
:???????????
while/lstm_cell_2/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_2/split:output:3*
T0*(
_output_shapes
:??????????e
#while/lstm_cell_2/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
(while/lstm_cell_2/split_1/ReadVariableOpReadVariableOp3while_lstm_cell_2_split_1_readvariableop_resource_0*
_output_shapes	
:?*
dtype0?
while/lstm_cell_2/split_1Split,while/lstm_cell_2/split_1/split_dim:output:00while/lstm_cell_2/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split?
while/lstm_cell_2/BiasAddBiasAdd"while/lstm_cell_2/MatMul:product:0"while/lstm_cell_2/split_1:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_2/BiasAdd_1BiasAdd$while/lstm_cell_2/MatMul_1:product:0"while/lstm_cell_2/split_1:output:1*
T0*(
_output_shapes
:???????????
while/lstm_cell_2/BiasAdd_2BiasAdd$while/lstm_cell_2/MatMul_2:product:0"while/lstm_cell_2/split_1:output:2*
T0*(
_output_shapes
:???????????
while/lstm_cell_2/BiasAdd_3BiasAdd$while/lstm_cell_2/MatMul_3:product:0"while/lstm_cell_2/split_1:output:3*
T0*(
_output_shapes
:???????????
 while/lstm_cell_2/ReadVariableOpReadVariableOp+while_lstm_cell_2_readvariableop_resource_0* 
_output_shapes
:
??*
dtype0v
%while/lstm_cell_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        x
'while/lstm_cell_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       x
'while/lstm_cell_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
while/lstm_cell_2/strided_sliceStridedSlice(while/lstm_cell_2/ReadVariableOp:value:0.while/lstm_cell_2/strided_slice/stack:output:00while/lstm_cell_2/strided_slice/stack_1:output:00while/lstm_cell_2/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
while/lstm_cell_2/MatMul_4MatMulwhile_placeholder_2(while/lstm_cell_2/strided_slice:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_2/addAddV2"while/lstm_cell_2/BiasAdd:output:0$while/lstm_cell_2/MatMul_4:product:0*
T0*(
_output_shapes
:??????????\
while/lstm_cell_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>^
while/lstm_cell_2/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ??
while/lstm_cell_2/MulMulwhile/lstm_cell_2/add:z:0 while/lstm_cell_2/Const:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_2/Add_1AddV2while/lstm_cell_2/Mul:z:0"while/lstm_cell_2/Const_1:output:0*
T0*(
_output_shapes
:??????????n
)while/lstm_cell_2/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
'while/lstm_cell_2/clip_by_value/MinimumMinimumwhile/lstm_cell_2/Add_1:z:02while/lstm_cell_2/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:??????????f
!while/lstm_cell_2/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
while/lstm_cell_2/clip_by_valueMaximum+while/lstm_cell_2/clip_by_value/Minimum:z:0*while/lstm_cell_2/clip_by_value/y:output:0*
T0*(
_output_shapes
:???????????
"while/lstm_cell_2/ReadVariableOp_1ReadVariableOp+while_lstm_cell_2_readvariableop_resource_0* 
_output_shapes
:
??*
dtype0x
'while/lstm_cell_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       z
)while/lstm_cell_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       z
)while/lstm_cell_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
!while/lstm_cell_2/strided_slice_1StridedSlice*while/lstm_cell_2/ReadVariableOp_1:value:00while/lstm_cell_2/strided_slice_1/stack:output:02while/lstm_cell_2/strided_slice_1/stack_1:output:02while/lstm_cell_2/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
while/lstm_cell_2/MatMul_5MatMulwhile_placeholder_2*while/lstm_cell_2/strided_slice_1:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_2/add_2AddV2$while/lstm_cell_2/BiasAdd_1:output:0$while/lstm_cell_2/MatMul_5:product:0*
T0*(
_output_shapes
:??????????^
while/lstm_cell_2/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>^
while/lstm_cell_2/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ??
while/lstm_cell_2/Mul_1Mulwhile/lstm_cell_2/add_2:z:0"while/lstm_cell_2/Const_2:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_2/Add_3AddV2while/lstm_cell_2/Mul_1:z:0"while/lstm_cell_2/Const_3:output:0*
T0*(
_output_shapes
:??????????p
+while/lstm_cell_2/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
)while/lstm_cell_2/clip_by_value_1/MinimumMinimumwhile/lstm_cell_2/Add_3:z:04while/lstm_cell_2/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:??????????h
#while/lstm_cell_2/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
!while/lstm_cell_2/clip_by_value_1Maximum-while/lstm_cell_2/clip_by_value_1/Minimum:z:0,while/lstm_cell_2/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_2/mul_2Mul%while/lstm_cell_2/clip_by_value_1:z:0while_placeholder_3*
T0*(
_output_shapes
:???????????
"while/lstm_cell_2/ReadVariableOp_2ReadVariableOp+while_lstm_cell_2_readvariableop_resource_0* 
_output_shapes
:
??*
dtype0x
'while/lstm_cell_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       z
)while/lstm_cell_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       z
)while/lstm_cell_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
!while/lstm_cell_2/strided_slice_2StridedSlice*while/lstm_cell_2/ReadVariableOp_2:value:00while/lstm_cell_2/strided_slice_2/stack:output:02while/lstm_cell_2/strided_slice_2/stack_1:output:02while/lstm_cell_2/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
while/lstm_cell_2/MatMul_6MatMulwhile_placeholder_2*while/lstm_cell_2/strided_slice_2:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_2/add_4AddV2$while/lstm_cell_2/BiasAdd_2:output:0$while/lstm_cell_2/MatMul_6:product:0*
T0*(
_output_shapes
:??????????n
while/lstm_cell_2/TanhTanhwhile/lstm_cell_2/add_4:z:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_2/mul_3Mul#while/lstm_cell_2/clip_by_value:z:0while/lstm_cell_2/Tanh:y:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_2/add_5AddV2while/lstm_cell_2/mul_2:z:0while/lstm_cell_2/mul_3:z:0*
T0*(
_output_shapes
:???????????
"while/lstm_cell_2/ReadVariableOp_3ReadVariableOp+while_lstm_cell_2_readvariableop_resource_0* 
_output_shapes
:
??*
dtype0x
'while/lstm_cell_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       z
)while/lstm_cell_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        z
)while/lstm_cell_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
!while/lstm_cell_2/strided_slice_3StridedSlice*while/lstm_cell_2/ReadVariableOp_3:value:00while/lstm_cell_2/strided_slice_3/stack:output:02while/lstm_cell_2/strided_slice_3/stack_1:output:02while/lstm_cell_2/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
while/lstm_cell_2/MatMul_7MatMulwhile_placeholder_2*while/lstm_cell_2/strided_slice_3:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_2/add_6AddV2$while/lstm_cell_2/BiasAdd_3:output:0$while/lstm_cell_2/MatMul_7:product:0*
T0*(
_output_shapes
:??????????^
while/lstm_cell_2/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>^
while/lstm_cell_2/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ??
while/lstm_cell_2/Mul_4Mulwhile/lstm_cell_2/add_6:z:0"while/lstm_cell_2/Const_4:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_2/Add_7AddV2while/lstm_cell_2/Mul_4:z:0"while/lstm_cell_2/Const_5:output:0*
T0*(
_output_shapes
:??????????p
+while/lstm_cell_2/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
)while/lstm_cell_2/clip_by_value_2/MinimumMinimumwhile/lstm_cell_2/Add_7:z:04while/lstm_cell_2/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:??????????h
#while/lstm_cell_2/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
!while/lstm_cell_2/clip_by_value_2Maximum-while/lstm_cell_2/clip_by_value_2/Minimum:z:0,while/lstm_cell_2/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:??????????p
while/lstm_cell_2/Tanh_1Tanhwhile/lstm_cell_2/add_5:z:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_2/mul_5Mul%while/lstm_cell_2/clip_by_value_2:z:0while/lstm_cell_2/Tanh_1:y:0*
T0*(
_output_shapes
:???????????
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_2/mul_5:z:0*
_output_shapes
: *
element_dtype0:???M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: y
while/Identity_4Identitywhile/lstm_cell_2/mul_5:z:0^while/NoOp*
T0*(
_output_shapes
:??????????y
while/Identity_5Identitywhile/lstm_cell_2/add_5:z:0^while/NoOp*
T0*(
_output_shapes
:???????????

while/NoOpNoOp!^while/lstm_cell_2/ReadVariableOp#^while/lstm_cell_2/ReadVariableOp_1#^while/lstm_cell_2/ReadVariableOp_2#^while/lstm_cell_2/ReadVariableOp_3'^while/lstm_cell_2/split/ReadVariableOp)^while/lstm_cell_2/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"X
)while_lstm_cell_2_readvariableop_resource+while_lstm_cell_2_readvariableop_resource_0"h
1while_lstm_cell_2_split_1_readvariableop_resource3while_lstm_cell_2_split_1_readvariableop_resource_0"d
/while_lstm_cell_2_split_readvariableop_resource1while_lstm_cell_2_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2D
 while/lstm_cell_2/ReadVariableOp while/lstm_cell_2/ReadVariableOp2H
"while/lstm_cell_2/ReadVariableOp_1"while/lstm_cell_2/ReadVariableOp_12H
"while/lstm_cell_2/ReadVariableOp_2"while/lstm_cell_2/ReadVariableOp_22H
"while/lstm_cell_2/ReadVariableOp_3"while/lstm_cell_2/ReadVariableOp_32P
&while/lstm_cell_2/split/ReadVariableOp&while/lstm_cell_2/split/ReadVariableOp2T
(while/lstm_cell_2/split_1/ReadVariableOp(while/lstm_cell_2/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?{
?	
while_body_55253
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
1while_lstm_cell_3_split_readvariableop_resource_0:
??B
3while_lstm_cell_3_split_1_readvariableop_resource_0:	??
+while_lstm_cell_3_readvariableop_resource_0:
??
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
/while_lstm_cell_3_split_readvariableop_resource:
??@
1while_lstm_cell_3_split_1_readvariableop_resource:	?=
)while_lstm_cell_3_readvariableop_resource:
???? while/lstm_cell_3/ReadVariableOp?"while/lstm_cell_3/ReadVariableOp_1?"while/lstm_cell_3/ReadVariableOp_2?"while/lstm_cell_3/ReadVariableOp_3?&while/lstm_cell_3/split/ReadVariableOp?(while/lstm_cell_3/split_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype0c
!while/lstm_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
&while/lstm_cell_3/split/ReadVariableOpReadVariableOp1while_lstm_cell_3_split_readvariableop_resource_0* 
_output_shapes
:
??*
dtype0?
while/lstm_cell_3/splitSplit*while/lstm_cell_3/split/split_dim:output:0.while/lstm_cell_3/split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
??:
??:
??:
??*
	num_split?
while/lstm_cell_3/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_3/split:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_3/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_3/split:output:1*
T0*(
_output_shapes
:???????????
while/lstm_cell_3/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_3/split:output:2*
T0*(
_output_shapes
:???????????
while/lstm_cell_3/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_3/split:output:3*
T0*(
_output_shapes
:??????????e
#while/lstm_cell_3/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
(while/lstm_cell_3/split_1/ReadVariableOpReadVariableOp3while_lstm_cell_3_split_1_readvariableop_resource_0*
_output_shapes	
:?*
dtype0?
while/lstm_cell_3/split_1Split,while/lstm_cell_3/split_1/split_dim:output:00while/lstm_cell_3/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split?
while/lstm_cell_3/BiasAddBiasAdd"while/lstm_cell_3/MatMul:product:0"while/lstm_cell_3/split_1:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_3/BiasAdd_1BiasAdd$while/lstm_cell_3/MatMul_1:product:0"while/lstm_cell_3/split_1:output:1*
T0*(
_output_shapes
:???????????
while/lstm_cell_3/BiasAdd_2BiasAdd$while/lstm_cell_3/MatMul_2:product:0"while/lstm_cell_3/split_1:output:2*
T0*(
_output_shapes
:???????????
while/lstm_cell_3/BiasAdd_3BiasAdd$while/lstm_cell_3/MatMul_3:product:0"while/lstm_cell_3/split_1:output:3*
T0*(
_output_shapes
:???????????
 while/lstm_cell_3/ReadVariableOpReadVariableOp+while_lstm_cell_3_readvariableop_resource_0* 
_output_shapes
:
??*
dtype0v
%while/lstm_cell_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        x
'while/lstm_cell_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   x
'while/lstm_cell_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
while/lstm_cell_3/strided_sliceStridedSlice(while/lstm_cell_3/ReadVariableOp:value:0.while/lstm_cell_3/strided_slice/stack:output:00while/lstm_cell_3/strided_slice/stack_1:output:00while/lstm_cell_3/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
while/lstm_cell_3/MatMul_4MatMulwhile_placeholder_2(while/lstm_cell_3/strided_slice:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_3/addAddV2"while/lstm_cell_3/BiasAdd:output:0$while/lstm_cell_3/MatMul_4:product:0*
T0*(
_output_shapes
:??????????\
while/lstm_cell_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>^
while/lstm_cell_3/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ??
while/lstm_cell_3/MulMulwhile/lstm_cell_3/add:z:0 while/lstm_cell_3/Const:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_3/Add_1AddV2while/lstm_cell_3/Mul:z:0"while/lstm_cell_3/Const_1:output:0*
T0*(
_output_shapes
:??????????n
)while/lstm_cell_3/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
'while/lstm_cell_3/clip_by_value/MinimumMinimumwhile/lstm_cell_3/Add_1:z:02while/lstm_cell_3/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:??????????f
!while/lstm_cell_3/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
while/lstm_cell_3/clip_by_valueMaximum+while/lstm_cell_3/clip_by_value/Minimum:z:0*while/lstm_cell_3/clip_by_value/y:output:0*
T0*(
_output_shapes
:???????????
"while/lstm_cell_3/ReadVariableOp_1ReadVariableOp+while_lstm_cell_3_readvariableop_resource_0* 
_output_shapes
:
??*
dtype0x
'while/lstm_cell_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   z
)while/lstm_cell_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       z
)while/lstm_cell_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
!while/lstm_cell_3/strided_slice_1StridedSlice*while/lstm_cell_3/ReadVariableOp_1:value:00while/lstm_cell_3/strided_slice_1/stack:output:02while/lstm_cell_3/strided_slice_1/stack_1:output:02while/lstm_cell_3/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
while/lstm_cell_3/MatMul_5MatMulwhile_placeholder_2*while/lstm_cell_3/strided_slice_1:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_3/add_2AddV2$while/lstm_cell_3/BiasAdd_1:output:0$while/lstm_cell_3/MatMul_5:product:0*
T0*(
_output_shapes
:??????????^
while/lstm_cell_3/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>^
while/lstm_cell_3/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ??
while/lstm_cell_3/Mul_1Mulwhile/lstm_cell_3/add_2:z:0"while/lstm_cell_3/Const_2:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_3/Add_3AddV2while/lstm_cell_3/Mul_1:z:0"while/lstm_cell_3/Const_3:output:0*
T0*(
_output_shapes
:??????????p
+while/lstm_cell_3/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
)while/lstm_cell_3/clip_by_value_1/MinimumMinimumwhile/lstm_cell_3/Add_3:z:04while/lstm_cell_3/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:??????????h
#while/lstm_cell_3/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
!while/lstm_cell_3/clip_by_value_1Maximum-while/lstm_cell_3/clip_by_value_1/Minimum:z:0,while/lstm_cell_3/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_3/mul_2Mul%while/lstm_cell_3/clip_by_value_1:z:0while_placeholder_3*
T0*(
_output_shapes
:???????????
"while/lstm_cell_3/ReadVariableOp_2ReadVariableOp+while_lstm_cell_3_readvariableop_resource_0* 
_output_shapes
:
??*
dtype0x
'while/lstm_cell_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       z
)while/lstm_cell_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  z
)while/lstm_cell_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
!while/lstm_cell_3/strided_slice_2StridedSlice*while/lstm_cell_3/ReadVariableOp_2:value:00while/lstm_cell_3/strided_slice_2/stack:output:02while/lstm_cell_3/strided_slice_2/stack_1:output:02while/lstm_cell_3/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
while/lstm_cell_3/MatMul_6MatMulwhile_placeholder_2*while/lstm_cell_3/strided_slice_2:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_3/add_4AddV2$while/lstm_cell_3/BiasAdd_2:output:0$while/lstm_cell_3/MatMul_6:product:0*
T0*(
_output_shapes
:??????????n
while/lstm_cell_3/TanhTanhwhile/lstm_cell_3/add_4:z:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_3/mul_3Mul#while/lstm_cell_3/clip_by_value:z:0while/lstm_cell_3/Tanh:y:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_3/add_5AddV2while/lstm_cell_3/mul_2:z:0while/lstm_cell_3/mul_3:z:0*
T0*(
_output_shapes
:???????????
"while/lstm_cell_3/ReadVariableOp_3ReadVariableOp+while_lstm_cell_3_readvariableop_resource_0* 
_output_shapes
:
??*
dtype0x
'while/lstm_cell_3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  z
)while/lstm_cell_3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        z
)while/lstm_cell_3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
!while/lstm_cell_3/strided_slice_3StridedSlice*while/lstm_cell_3/ReadVariableOp_3:value:00while/lstm_cell_3/strided_slice_3/stack:output:02while/lstm_cell_3/strided_slice_3/stack_1:output:02while/lstm_cell_3/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
while/lstm_cell_3/MatMul_7MatMulwhile_placeholder_2*while/lstm_cell_3/strided_slice_3:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_3/add_6AddV2$while/lstm_cell_3/BiasAdd_3:output:0$while/lstm_cell_3/MatMul_7:product:0*
T0*(
_output_shapes
:??????????^
while/lstm_cell_3/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>^
while/lstm_cell_3/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ??
while/lstm_cell_3/Mul_4Mulwhile/lstm_cell_3/add_6:z:0"while/lstm_cell_3/Const_4:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_3/Add_7AddV2while/lstm_cell_3/Mul_4:z:0"while/lstm_cell_3/Const_5:output:0*
T0*(
_output_shapes
:??????????p
+while/lstm_cell_3/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
)while/lstm_cell_3/clip_by_value_2/MinimumMinimumwhile/lstm_cell_3/Add_7:z:04while/lstm_cell_3/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:??????????h
#while/lstm_cell_3/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
!while/lstm_cell_3/clip_by_value_2Maximum-while/lstm_cell_3/clip_by_value_2/Minimum:z:0,while/lstm_cell_3/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:??????????p
while/lstm_cell_3/Tanh_1Tanhwhile/lstm_cell_3/add_5:z:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_3/mul_5Mul%while/lstm_cell_3/clip_by_value_2:z:0while/lstm_cell_3/Tanh_1:y:0*
T0*(
_output_shapes
:???????????
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_3/mul_5:z:0*
_output_shapes
: *
element_dtype0:???M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: y
while/Identity_4Identitywhile/lstm_cell_3/mul_5:z:0^while/NoOp*
T0*(
_output_shapes
:??????????y
while/Identity_5Identitywhile/lstm_cell_3/add_5:z:0^while/NoOp*
T0*(
_output_shapes
:???????????

while/NoOpNoOp!^while/lstm_cell_3/ReadVariableOp#^while/lstm_cell_3/ReadVariableOp_1#^while/lstm_cell_3/ReadVariableOp_2#^while/lstm_cell_3/ReadVariableOp_3'^while/lstm_cell_3/split/ReadVariableOp)^while/lstm_cell_3/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"X
)while_lstm_cell_3_readvariableop_resource+while_lstm_cell_3_readvariableop_resource_0"h
1while_lstm_cell_3_split_1_readvariableop_resource3while_lstm_cell_3_split_1_readvariableop_resource_0"d
/while_lstm_cell_3_split_readvariableop_resource1while_lstm_cell_3_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2D
 while/lstm_cell_3/ReadVariableOp while/lstm_cell_3/ReadVariableOp2H
"while/lstm_cell_3/ReadVariableOp_1"while/lstm_cell_3/ReadVariableOp_12H
"while/lstm_cell_3/ReadVariableOp_2"while/lstm_cell_3/ReadVariableOp_22H
"while/lstm_cell_3/ReadVariableOp_3"while/lstm_cell_3/ReadVariableOp_32P
&while/lstm_cell_3/split/ReadVariableOp&while/lstm_cell_3/split/ReadVariableOp2T
(while/lstm_cell_3/split_1/ReadVariableOp(while/lstm_cell_3/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?{
?	
while_body_59273
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
1while_lstm_cell_3_split_readvariableop_resource_0:
??B
3while_lstm_cell_3_split_1_readvariableop_resource_0:	??
+while_lstm_cell_3_readvariableop_resource_0:
??
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
/while_lstm_cell_3_split_readvariableop_resource:
??@
1while_lstm_cell_3_split_1_readvariableop_resource:	?=
)while_lstm_cell_3_readvariableop_resource:
???? while/lstm_cell_3/ReadVariableOp?"while/lstm_cell_3/ReadVariableOp_1?"while/lstm_cell_3/ReadVariableOp_2?"while/lstm_cell_3/ReadVariableOp_3?&while/lstm_cell_3/split/ReadVariableOp?(while/lstm_cell_3/split_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype0c
!while/lstm_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
&while/lstm_cell_3/split/ReadVariableOpReadVariableOp1while_lstm_cell_3_split_readvariableop_resource_0* 
_output_shapes
:
??*
dtype0?
while/lstm_cell_3/splitSplit*while/lstm_cell_3/split/split_dim:output:0.while/lstm_cell_3/split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
??:
??:
??:
??*
	num_split?
while/lstm_cell_3/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_3/split:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_3/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_3/split:output:1*
T0*(
_output_shapes
:???????????
while/lstm_cell_3/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_3/split:output:2*
T0*(
_output_shapes
:???????????
while/lstm_cell_3/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_3/split:output:3*
T0*(
_output_shapes
:??????????e
#while/lstm_cell_3/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
(while/lstm_cell_3/split_1/ReadVariableOpReadVariableOp3while_lstm_cell_3_split_1_readvariableop_resource_0*
_output_shapes	
:?*
dtype0?
while/lstm_cell_3/split_1Split,while/lstm_cell_3/split_1/split_dim:output:00while/lstm_cell_3/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split?
while/lstm_cell_3/BiasAddBiasAdd"while/lstm_cell_3/MatMul:product:0"while/lstm_cell_3/split_1:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_3/BiasAdd_1BiasAdd$while/lstm_cell_3/MatMul_1:product:0"while/lstm_cell_3/split_1:output:1*
T0*(
_output_shapes
:???????????
while/lstm_cell_3/BiasAdd_2BiasAdd$while/lstm_cell_3/MatMul_2:product:0"while/lstm_cell_3/split_1:output:2*
T0*(
_output_shapes
:???????????
while/lstm_cell_3/BiasAdd_3BiasAdd$while/lstm_cell_3/MatMul_3:product:0"while/lstm_cell_3/split_1:output:3*
T0*(
_output_shapes
:???????????
 while/lstm_cell_3/ReadVariableOpReadVariableOp+while_lstm_cell_3_readvariableop_resource_0* 
_output_shapes
:
??*
dtype0v
%while/lstm_cell_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        x
'while/lstm_cell_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   x
'while/lstm_cell_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
while/lstm_cell_3/strided_sliceStridedSlice(while/lstm_cell_3/ReadVariableOp:value:0.while/lstm_cell_3/strided_slice/stack:output:00while/lstm_cell_3/strided_slice/stack_1:output:00while/lstm_cell_3/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
while/lstm_cell_3/MatMul_4MatMulwhile_placeholder_2(while/lstm_cell_3/strided_slice:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_3/addAddV2"while/lstm_cell_3/BiasAdd:output:0$while/lstm_cell_3/MatMul_4:product:0*
T0*(
_output_shapes
:??????????\
while/lstm_cell_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>^
while/lstm_cell_3/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ??
while/lstm_cell_3/MulMulwhile/lstm_cell_3/add:z:0 while/lstm_cell_3/Const:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_3/Add_1AddV2while/lstm_cell_3/Mul:z:0"while/lstm_cell_3/Const_1:output:0*
T0*(
_output_shapes
:??????????n
)while/lstm_cell_3/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
'while/lstm_cell_3/clip_by_value/MinimumMinimumwhile/lstm_cell_3/Add_1:z:02while/lstm_cell_3/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:??????????f
!while/lstm_cell_3/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
while/lstm_cell_3/clip_by_valueMaximum+while/lstm_cell_3/clip_by_value/Minimum:z:0*while/lstm_cell_3/clip_by_value/y:output:0*
T0*(
_output_shapes
:???????????
"while/lstm_cell_3/ReadVariableOp_1ReadVariableOp+while_lstm_cell_3_readvariableop_resource_0* 
_output_shapes
:
??*
dtype0x
'while/lstm_cell_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   z
)while/lstm_cell_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       z
)while/lstm_cell_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
!while/lstm_cell_3/strided_slice_1StridedSlice*while/lstm_cell_3/ReadVariableOp_1:value:00while/lstm_cell_3/strided_slice_1/stack:output:02while/lstm_cell_3/strided_slice_1/stack_1:output:02while/lstm_cell_3/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
while/lstm_cell_3/MatMul_5MatMulwhile_placeholder_2*while/lstm_cell_3/strided_slice_1:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_3/add_2AddV2$while/lstm_cell_3/BiasAdd_1:output:0$while/lstm_cell_3/MatMul_5:product:0*
T0*(
_output_shapes
:??????????^
while/lstm_cell_3/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>^
while/lstm_cell_3/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ??
while/lstm_cell_3/Mul_1Mulwhile/lstm_cell_3/add_2:z:0"while/lstm_cell_3/Const_2:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_3/Add_3AddV2while/lstm_cell_3/Mul_1:z:0"while/lstm_cell_3/Const_3:output:0*
T0*(
_output_shapes
:??????????p
+while/lstm_cell_3/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
)while/lstm_cell_3/clip_by_value_1/MinimumMinimumwhile/lstm_cell_3/Add_3:z:04while/lstm_cell_3/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:??????????h
#while/lstm_cell_3/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
!while/lstm_cell_3/clip_by_value_1Maximum-while/lstm_cell_3/clip_by_value_1/Minimum:z:0,while/lstm_cell_3/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_3/mul_2Mul%while/lstm_cell_3/clip_by_value_1:z:0while_placeholder_3*
T0*(
_output_shapes
:???????????
"while/lstm_cell_3/ReadVariableOp_2ReadVariableOp+while_lstm_cell_3_readvariableop_resource_0* 
_output_shapes
:
??*
dtype0x
'while/lstm_cell_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       z
)while/lstm_cell_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  z
)while/lstm_cell_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
!while/lstm_cell_3/strided_slice_2StridedSlice*while/lstm_cell_3/ReadVariableOp_2:value:00while/lstm_cell_3/strided_slice_2/stack:output:02while/lstm_cell_3/strided_slice_2/stack_1:output:02while/lstm_cell_3/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
while/lstm_cell_3/MatMul_6MatMulwhile_placeholder_2*while/lstm_cell_3/strided_slice_2:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_3/add_4AddV2$while/lstm_cell_3/BiasAdd_2:output:0$while/lstm_cell_3/MatMul_6:product:0*
T0*(
_output_shapes
:??????????n
while/lstm_cell_3/TanhTanhwhile/lstm_cell_3/add_4:z:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_3/mul_3Mul#while/lstm_cell_3/clip_by_value:z:0while/lstm_cell_3/Tanh:y:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_3/add_5AddV2while/lstm_cell_3/mul_2:z:0while/lstm_cell_3/mul_3:z:0*
T0*(
_output_shapes
:???????????
"while/lstm_cell_3/ReadVariableOp_3ReadVariableOp+while_lstm_cell_3_readvariableop_resource_0* 
_output_shapes
:
??*
dtype0x
'while/lstm_cell_3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  z
)while/lstm_cell_3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        z
)while/lstm_cell_3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
!while/lstm_cell_3/strided_slice_3StridedSlice*while/lstm_cell_3/ReadVariableOp_3:value:00while/lstm_cell_3/strided_slice_3/stack:output:02while/lstm_cell_3/strided_slice_3/stack_1:output:02while/lstm_cell_3/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
while/lstm_cell_3/MatMul_7MatMulwhile_placeholder_2*while/lstm_cell_3/strided_slice_3:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_3/add_6AddV2$while/lstm_cell_3/BiasAdd_3:output:0$while/lstm_cell_3/MatMul_7:product:0*
T0*(
_output_shapes
:??????????^
while/lstm_cell_3/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>^
while/lstm_cell_3/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ??
while/lstm_cell_3/Mul_4Mulwhile/lstm_cell_3/add_6:z:0"while/lstm_cell_3/Const_4:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_3/Add_7AddV2while/lstm_cell_3/Mul_4:z:0"while/lstm_cell_3/Const_5:output:0*
T0*(
_output_shapes
:??????????p
+while/lstm_cell_3/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
)while/lstm_cell_3/clip_by_value_2/MinimumMinimumwhile/lstm_cell_3/Add_7:z:04while/lstm_cell_3/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:??????????h
#while/lstm_cell_3/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
!while/lstm_cell_3/clip_by_value_2Maximum-while/lstm_cell_3/clip_by_value_2/Minimum:z:0,while/lstm_cell_3/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:??????????p
while/lstm_cell_3/Tanh_1Tanhwhile/lstm_cell_3/add_5:z:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_3/mul_5Mul%while/lstm_cell_3/clip_by_value_2:z:0while/lstm_cell_3/Tanh_1:y:0*
T0*(
_output_shapes
:???????????
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_3/mul_5:z:0*
_output_shapes
: *
element_dtype0:???M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: y
while/Identity_4Identitywhile/lstm_cell_3/mul_5:z:0^while/NoOp*
T0*(
_output_shapes
:??????????y
while/Identity_5Identitywhile/lstm_cell_3/add_5:z:0^while/NoOp*
T0*(
_output_shapes
:???????????

while/NoOpNoOp!^while/lstm_cell_3/ReadVariableOp#^while/lstm_cell_3/ReadVariableOp_1#^while/lstm_cell_3/ReadVariableOp_2#^while/lstm_cell_3/ReadVariableOp_3'^while/lstm_cell_3/split/ReadVariableOp)^while/lstm_cell_3/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"X
)while_lstm_cell_3_readvariableop_resource+while_lstm_cell_3_readvariableop_resource_0"h
1while_lstm_cell_3_split_1_readvariableop_resource3while_lstm_cell_3_split_1_readvariableop_resource_0"d
/while_lstm_cell_3_split_readvariableop_resource1while_lstm_cell_3_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2D
 while/lstm_cell_3/ReadVariableOp while/lstm_cell_3/ReadVariableOp2H
"while/lstm_cell_3/ReadVariableOp_1"while/lstm_cell_3/ReadVariableOp_12H
"while/lstm_cell_3/ReadVariableOp_2"while/lstm_cell_3/ReadVariableOp_22H
"while/lstm_cell_3/ReadVariableOp_3"while/lstm_cell_3/ReadVariableOp_32P
&while/lstm_cell_3/split/ReadVariableOp&while/lstm_cell_3/split/ReadVariableOp2T
(while/lstm_cell_3/split_1/ReadVariableOp(while/lstm_cell_3/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?R
?
__inference__traced_save_60061
file_prefix-
)savev2_dense_2_kernel_read_readvariableop+
'savev2_dense_2_bias_read_readvariableop-
)savev2_dense_3_kernel_read_readvariableop+
'savev2_dense_3_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop8
4savev2_lstm_2_lstm_cell_2_kernel_read_readvariableopB
>savev2_lstm_2_lstm_cell_2_recurrent_kernel_read_readvariableop6
2savev2_lstm_2_lstm_cell_2_bias_read_readvariableop8
4savev2_lstm_3_lstm_cell_3_kernel_read_readvariableopB
>savev2_lstm_3_lstm_cell_3_recurrent_kernel_read_readvariableop6
2savev2_lstm_3_lstm_cell_3_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop4
0savev2_adam_dense_2_kernel_m_read_readvariableop2
.savev2_adam_dense_2_bias_m_read_readvariableop4
0savev2_adam_dense_3_kernel_m_read_readvariableop2
.savev2_adam_dense_3_bias_m_read_readvariableop?
;savev2_adam_lstm_2_lstm_cell_2_kernel_m_read_readvariableopI
Esavev2_adam_lstm_2_lstm_cell_2_recurrent_kernel_m_read_readvariableop=
9savev2_adam_lstm_2_lstm_cell_2_bias_m_read_readvariableop?
;savev2_adam_lstm_3_lstm_cell_3_kernel_m_read_readvariableopI
Esavev2_adam_lstm_3_lstm_cell_3_recurrent_kernel_m_read_readvariableop=
9savev2_adam_lstm_3_lstm_cell_3_bias_m_read_readvariableop4
0savev2_adam_dense_2_kernel_v_read_readvariableop2
.savev2_adam_dense_2_bias_v_read_readvariableop4
0savev2_adam_dense_3_kernel_v_read_readvariableop2
.savev2_adam_dense_3_bias_v_read_readvariableop?
;savev2_adam_lstm_2_lstm_cell_2_kernel_v_read_readvariableopI
Esavev2_adam_lstm_2_lstm_cell_2_recurrent_kernel_v_read_readvariableop=
9savev2_adam_lstm_2_lstm_cell_2_bias_v_read_readvariableop?
;savev2_adam_lstm_3_lstm_cell_3_kernel_v_read_readvariableopI
Esavev2_adam_lstm_3_lstm_cell_3_recurrent_kernel_v_read_readvariableop=
9savev2_adam_lstm_3_lstm_cell_3_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : ?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:(*
dtype0*?
value?B?(B6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:(*
dtype0*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop)savev2_dense_3_kernel_read_readvariableop'savev2_dense_3_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop4savev2_lstm_2_lstm_cell_2_kernel_read_readvariableop>savev2_lstm_2_lstm_cell_2_recurrent_kernel_read_readvariableop2savev2_lstm_2_lstm_cell_2_bias_read_readvariableop4savev2_lstm_3_lstm_cell_3_kernel_read_readvariableop>savev2_lstm_3_lstm_cell_3_recurrent_kernel_read_readvariableop2savev2_lstm_3_lstm_cell_3_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop0savev2_adam_dense_2_kernel_m_read_readvariableop.savev2_adam_dense_2_bias_m_read_readvariableop0savev2_adam_dense_3_kernel_m_read_readvariableop.savev2_adam_dense_3_bias_m_read_readvariableop;savev2_adam_lstm_2_lstm_cell_2_kernel_m_read_readvariableopEsavev2_adam_lstm_2_lstm_cell_2_recurrent_kernel_m_read_readvariableop9savev2_adam_lstm_2_lstm_cell_2_bias_m_read_readvariableop;savev2_adam_lstm_3_lstm_cell_3_kernel_m_read_readvariableopEsavev2_adam_lstm_3_lstm_cell_3_recurrent_kernel_m_read_readvariableop9savev2_adam_lstm_3_lstm_cell_3_bias_m_read_readvariableop0savev2_adam_dense_2_kernel_v_read_readvariableop.savev2_adam_dense_2_bias_v_read_readvariableop0savev2_adam_dense_3_kernel_v_read_readvariableop.savev2_adam_dense_3_bias_v_read_readvariableop;savev2_adam_lstm_2_lstm_cell_2_kernel_v_read_readvariableopEsavev2_adam_lstm_2_lstm_cell_2_recurrent_kernel_v_read_readvariableop9savev2_adam_lstm_2_lstm_cell_2_bias_v_read_readvariableop;savev2_adam_lstm_3_lstm_cell_3_kernel_v_read_readvariableopEsavev2_adam_lstm_3_lstm_cell_3_recurrent_kernel_v_read_readvariableop9savev2_adam_lstm_3_lstm_cell_3_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *6
dtypes,
*2(	?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*?
_input_shapes?
?: :
??:?:	?:: : : : : :	?:
??:?:
??:
??:?: : : : :
??:?:	?::	?:
??:?:
??:
??:?:
??:?:	?::	?:
??:?:
??:
??:?: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:%!

_output_shapes
:	?: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :%
!

_output_shapes
:	?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :&"
 
_output_shapes
:
??:!

_output_shapes	
:?:%!

_output_shapes
:	?: 

_output_shapes
::%!

_output_shapes
:	?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:% !

_output_shapes
:	?: !

_output_shapes
::%"!

_output_shapes
:	?:&#"
 
_output_shapes
:
??:!$

_output_shapes	
:?:&%"
 
_output_shapes
:
??:&&"
 
_output_shapes
:
??:!'

_output_shapes	
:?:(

_output_shapes
: 
?K
?
F__inference_lstm_cell_2_layer_call_and_return_conditional_losses_59675

inputs
states_0
states_10
split_readvariableop_resource:	?.
split_1_readvariableop_resource:	?+
readvariableop_resource:
??
identity

identity_1

identity_2??ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?ReadVariableOp_3?split/ReadVariableOp?split_1/ReadVariableOpQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :s
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes
:	?*
dtype0?
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	?:	?:	?:	?*
	num_split[
MatMulMatMulinputssplit:output:0*
T0*(
_output_shapes
:??????????]
MatMul_1MatMulinputssplit:output:1*
T0*(
_output_shapes
:??????????]
MatMul_2MatMulinputssplit:output:2*
T0*(
_output_shapes
:??????????]
MatMul_3MatMulinputssplit:output:3*
T0*(
_output_shapes
:??????????S
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : s
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes	
:?*
dtype0?
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_spliti
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*(
_output_shapes
:??????????m
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*(
_output_shapes
:??????????m
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*(
_output_shapes
:??????????m
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*(
_output_shapes
:??????????h
ReadVariableOpReadVariableOpreadvariableop_resource* 
_output_shapes
:
??*
dtype0d
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_maskg
MatMul_4MatMulstates_0strided_slice:output:0*
T0*(
_output_shapes
:??????????e
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*(
_output_shapes
:??????????J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>L
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?V
MulMuladd:z:0Const:output:0*
T0*(
_output_shapes
:??????????\
Add_1AddV2Mul:z:0Const_1:output:0*
T0*(
_output_shapes
:??????????\
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:??????????T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*(
_output_shapes
:??????????j
ReadVariableOp_1ReadVariableOpreadvariableop_resource* 
_output_shapes
:
??*
dtype0f
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_maski
MatMul_5MatMulstates_0strided_slice_1:output:0*
T0*(
_output_shapes
:??????????i
add_2AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*(
_output_shapes
:??????????L
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>L
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?\
Mul_1Mul	add_2:z:0Const_2:output:0*
T0*(
_output_shapes
:??????????^
Add_3AddV2	Mul_1:z:0Const_3:output:0*
T0*(
_output_shapes
:??????????^
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:??????????V
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*(
_output_shapes
:??????????^
mul_2Mulclip_by_value_1:z:0states_1*
T0*(
_output_shapes
:??????????j
ReadVariableOp_2ReadVariableOpreadvariableop_resource* 
_output_shapes
:
??*
dtype0f
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_maski
MatMul_6MatMulstates_0strided_slice_2:output:0*
T0*(
_output_shapes
:??????????i
add_4AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*(
_output_shapes
:??????????J
TanhTanh	add_4:z:0*
T0*(
_output_shapes
:??????????\
mul_3Mulclip_by_value:z:0Tanh:y:0*
T0*(
_output_shapes
:??????????W
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*(
_output_shapes
:??????????j
ReadVariableOp_3ReadVariableOpreadvariableop_resource* 
_output_shapes
:
??*
dtype0f
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_maski
MatMul_7MatMulstates_0strided_slice_3:output:0*
T0*(
_output_shapes
:??????????i
add_6AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*(
_output_shapes
:??????????L
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>L
Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?\
Mul_4Mul	add_6:z:0Const_4:output:0*
T0*(
_output_shapes
:??????????^
Add_7AddV2	Mul_4:z:0Const_5:output:0*
T0*(
_output_shapes
:??????????^
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:??????????V
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*(
_output_shapes
:??????????L
Tanh_1Tanh	add_5:z:0*
T0*(
_output_shapes
:??????????`
mul_5Mulclip_by_value_2:z:0
Tanh_1:y:0*
T0*(
_output_shapes
:??????????Y
IdentityIdentity	mul_5:z:0^NoOp*
T0*(
_output_shapes
:??????????[

Identity_1Identity	mul_5:z:0^NoOp*
T0*(
_output_shapes
:??????????[

Identity_2Identity	add_5:z:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:?????????:??????????:??????????: : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/0:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/1
?7
?
A__inference_lstm_3_layer_call_and_return_conditional_losses_54859

inputs%
lstm_cell_3_54778:
?? 
lstm_cell_3_54780:	?%
lstm_cell_3_54782:
??
identity??#lstm_cell_3/StatefulPartitionedCall?while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????S
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          w
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:???????????????????D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask?
#lstm_cell_3/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_3_54778lstm_cell_3_54780lstm_cell_3_54782*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:??????????:??????????:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_lstm_cell_3_layer_call_and_return_conditional_losses_54732n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   ?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_3_54778lstm_cell_3_54780lstm_cell_3_54782*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_54791*
condR
while_cond_54790*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:???????????????????*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:???????????????????h
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*(
_output_shapes
:??????????t
NoOpNoOp$^lstm_cell_3/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':???????????????????: : : 2J
#lstm_cell_3/StatefulPartitionedCall#lstm_cell_3/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs
??
?
A__inference_lstm_2_layer_call_and_return_conditional_losses_58345

inputs<
)lstm_cell_2_split_readvariableop_resource:	?:
+lstm_cell_2_split_1_readvariableop_resource:	?7
#lstm_cell_2_readvariableop_resource:
??
identity??lstm_cell_2/ReadVariableOp?lstm_cell_2/ReadVariableOp_1?lstm_cell_2/ReadVariableOp_2?lstm_cell_2/ReadVariableOp_3? lstm_cell_2/split/ReadVariableOp?"lstm_cell_2/split_1/ReadVariableOp?while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????S
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:?????????D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask]
lstm_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
 lstm_cell_2/split/ReadVariableOpReadVariableOp)lstm_cell_2_split_readvariableop_resource*
_output_shapes
:	?*
dtype0?
lstm_cell_2/splitSplit$lstm_cell_2/split/split_dim:output:0(lstm_cell_2/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	?:	?:	?:	?*
	num_split?
lstm_cell_2/MatMulMatMulstrided_slice_2:output:0lstm_cell_2/split:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_2/MatMul_1MatMulstrided_slice_2:output:0lstm_cell_2/split:output:1*
T0*(
_output_shapes
:???????????
lstm_cell_2/MatMul_2MatMulstrided_slice_2:output:0lstm_cell_2/split:output:2*
T0*(
_output_shapes
:???????????
lstm_cell_2/MatMul_3MatMulstrided_slice_2:output:0lstm_cell_2/split:output:3*
T0*(
_output_shapes
:??????????_
lstm_cell_2/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
"lstm_cell_2/split_1/ReadVariableOpReadVariableOp+lstm_cell_2_split_1_readvariableop_resource*
_output_shapes	
:?*
dtype0?
lstm_cell_2/split_1Split&lstm_cell_2/split_1/split_dim:output:0*lstm_cell_2/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split?
lstm_cell_2/BiasAddBiasAddlstm_cell_2/MatMul:product:0lstm_cell_2/split_1:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_2/BiasAdd_1BiasAddlstm_cell_2/MatMul_1:product:0lstm_cell_2/split_1:output:1*
T0*(
_output_shapes
:???????????
lstm_cell_2/BiasAdd_2BiasAddlstm_cell_2/MatMul_2:product:0lstm_cell_2/split_1:output:2*
T0*(
_output_shapes
:???????????
lstm_cell_2/BiasAdd_3BiasAddlstm_cell_2/MatMul_3:product:0lstm_cell_2/split_1:output:3*
T0*(
_output_shapes
:???????????
lstm_cell_2/ReadVariableOpReadVariableOp#lstm_cell_2_readvariableop_resource* 
_output_shapes
:
??*
dtype0p
lstm_cell_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        r
!lstm_cell_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       r
!lstm_cell_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
lstm_cell_2/strided_sliceStridedSlice"lstm_cell_2/ReadVariableOp:value:0(lstm_cell_2/strided_slice/stack:output:0*lstm_cell_2/strided_slice/stack_1:output:0*lstm_cell_2/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
lstm_cell_2/MatMul_4MatMulzeros:output:0"lstm_cell_2/strided_slice:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_2/addAddV2lstm_cell_2/BiasAdd:output:0lstm_cell_2/MatMul_4:product:0*
T0*(
_output_shapes
:??????????V
lstm_cell_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>X
lstm_cell_2/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?z
lstm_cell_2/MulMullstm_cell_2/add:z:0lstm_cell_2/Const:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_2/Add_1AddV2lstm_cell_2/Mul:z:0lstm_cell_2/Const_1:output:0*
T0*(
_output_shapes
:??????????h
#lstm_cell_2/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
!lstm_cell_2/clip_by_value/MinimumMinimumlstm_cell_2/Add_1:z:0,lstm_cell_2/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:??????????`
lstm_cell_2/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
lstm_cell_2/clip_by_valueMaximum%lstm_cell_2/clip_by_value/Minimum:z:0$lstm_cell_2/clip_by_value/y:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_2/ReadVariableOp_1ReadVariableOp#lstm_cell_2_readvariableop_resource* 
_output_shapes
:
??*
dtype0r
!lstm_cell_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       t
#lstm_cell_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       t
#lstm_cell_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
lstm_cell_2/strided_slice_1StridedSlice$lstm_cell_2/ReadVariableOp_1:value:0*lstm_cell_2/strided_slice_1/stack:output:0,lstm_cell_2/strided_slice_1/stack_1:output:0,lstm_cell_2/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
lstm_cell_2/MatMul_5MatMulzeros:output:0$lstm_cell_2/strided_slice_1:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_2/add_2AddV2lstm_cell_2/BiasAdd_1:output:0lstm_cell_2/MatMul_5:product:0*
T0*(
_output_shapes
:??????????X
lstm_cell_2/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>X
lstm_cell_2/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ??
lstm_cell_2/Mul_1Mullstm_cell_2/add_2:z:0lstm_cell_2/Const_2:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_2/Add_3AddV2lstm_cell_2/Mul_1:z:0lstm_cell_2/Const_3:output:0*
T0*(
_output_shapes
:??????????j
%lstm_cell_2/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
#lstm_cell_2/clip_by_value_1/MinimumMinimumlstm_cell_2/Add_3:z:0.lstm_cell_2/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:??????????b
lstm_cell_2/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
lstm_cell_2/clip_by_value_1Maximum'lstm_cell_2/clip_by_value_1/Minimum:z:0&lstm_cell_2/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:??????????~
lstm_cell_2/mul_2Mullstm_cell_2/clip_by_value_1:z:0zeros_1:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_2/ReadVariableOp_2ReadVariableOp#lstm_cell_2_readvariableop_resource* 
_output_shapes
:
??*
dtype0r
!lstm_cell_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       t
#lstm_cell_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       t
#lstm_cell_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
lstm_cell_2/strided_slice_2StridedSlice$lstm_cell_2/ReadVariableOp_2:value:0*lstm_cell_2/strided_slice_2/stack:output:0,lstm_cell_2/strided_slice_2/stack_1:output:0,lstm_cell_2/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
lstm_cell_2/MatMul_6MatMulzeros:output:0$lstm_cell_2/strided_slice_2:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_2/add_4AddV2lstm_cell_2/BiasAdd_2:output:0lstm_cell_2/MatMul_6:product:0*
T0*(
_output_shapes
:??????????b
lstm_cell_2/TanhTanhlstm_cell_2/add_4:z:0*
T0*(
_output_shapes
:???????????
lstm_cell_2/mul_3Mullstm_cell_2/clip_by_value:z:0lstm_cell_2/Tanh:y:0*
T0*(
_output_shapes
:??????????{
lstm_cell_2/add_5AddV2lstm_cell_2/mul_2:z:0lstm_cell_2/mul_3:z:0*
T0*(
_output_shapes
:???????????
lstm_cell_2/ReadVariableOp_3ReadVariableOp#lstm_cell_2_readvariableop_resource* 
_output_shapes
:
??*
dtype0r
!lstm_cell_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       t
#lstm_cell_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        t
#lstm_cell_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
lstm_cell_2/strided_slice_3StridedSlice$lstm_cell_2/ReadVariableOp_3:value:0*lstm_cell_2/strided_slice_3/stack:output:0,lstm_cell_2/strided_slice_3/stack_1:output:0,lstm_cell_2/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
lstm_cell_2/MatMul_7MatMulzeros:output:0$lstm_cell_2/strided_slice_3:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_2/add_6AddV2lstm_cell_2/BiasAdd_3:output:0lstm_cell_2/MatMul_7:product:0*
T0*(
_output_shapes
:??????????X
lstm_cell_2/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>X
lstm_cell_2/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ??
lstm_cell_2/Mul_4Mullstm_cell_2/add_6:z:0lstm_cell_2/Const_4:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_2/Add_7AddV2lstm_cell_2/Mul_4:z:0lstm_cell_2/Const_5:output:0*
T0*(
_output_shapes
:??????????j
%lstm_cell_2/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
#lstm_cell_2/clip_by_value_2/MinimumMinimumlstm_cell_2/Add_7:z:0.lstm_cell_2/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:??????????b
lstm_cell_2/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
lstm_cell_2/clip_by_value_2Maximum'lstm_cell_2/clip_by_value_2/Minimum:z:0&lstm_cell_2/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:??????????d
lstm_cell_2/Tanh_1Tanhlstm_cell_2/add_5:z:0*
T0*(
_output_shapes
:???????????
lstm_cell_2/mul_5Mullstm_cell_2/clip_by_value_2:z:0lstm_cell_2/Tanh_1:y:0*
T0*(
_output_shapes
:??????????n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)lstm_cell_2_split_readvariableop_resource+lstm_cell_2_split_1_readvariableop_resource#lstm_cell_2_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_58205*
condR
while_cond_58204*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????c
IdentityIdentitytranspose_1:y:0^NoOp*
T0*,
_output_shapes
:???????????
NoOpNoOp^lstm_cell_2/ReadVariableOp^lstm_cell_2/ReadVariableOp_1^lstm_cell_2/ReadVariableOp_2^lstm_cell_2/ReadVariableOp_3!^lstm_cell_2/split/ReadVariableOp#^lstm_cell_2/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????: : : 28
lstm_cell_2/ReadVariableOplstm_cell_2/ReadVariableOp2<
lstm_cell_2/ReadVariableOp_1lstm_cell_2/ReadVariableOp_12<
lstm_cell_2/ReadVariableOp_2lstm_cell_2/ReadVariableOp_22<
lstm_cell_2/ReadVariableOp_3lstm_cell_2/ReadVariableOp_32D
 lstm_cell_2/split/ReadVariableOp lstm_cell_2/split/ReadVariableOp2H
"lstm_cell_2/split_1/ReadVariableOp"lstm_cell_2/split_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
B__inference_dense_2_layer_call_and_return_conditional_losses_59468

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?{
?	
while_body_58761
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
1while_lstm_cell_3_split_readvariableop_resource_0:
??B
3while_lstm_cell_3_split_1_readvariableop_resource_0:	??
+while_lstm_cell_3_readvariableop_resource_0:
??
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
/while_lstm_cell_3_split_readvariableop_resource:
??@
1while_lstm_cell_3_split_1_readvariableop_resource:	?=
)while_lstm_cell_3_readvariableop_resource:
???? while/lstm_cell_3/ReadVariableOp?"while/lstm_cell_3/ReadVariableOp_1?"while/lstm_cell_3/ReadVariableOp_2?"while/lstm_cell_3/ReadVariableOp_3?&while/lstm_cell_3/split/ReadVariableOp?(while/lstm_cell_3/split_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype0c
!while/lstm_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
&while/lstm_cell_3/split/ReadVariableOpReadVariableOp1while_lstm_cell_3_split_readvariableop_resource_0* 
_output_shapes
:
??*
dtype0?
while/lstm_cell_3/splitSplit*while/lstm_cell_3/split/split_dim:output:0.while/lstm_cell_3/split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
??:
??:
??:
??*
	num_split?
while/lstm_cell_3/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_3/split:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_3/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_3/split:output:1*
T0*(
_output_shapes
:???????????
while/lstm_cell_3/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_3/split:output:2*
T0*(
_output_shapes
:???????????
while/lstm_cell_3/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_3/split:output:3*
T0*(
_output_shapes
:??????????e
#while/lstm_cell_3/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
(while/lstm_cell_3/split_1/ReadVariableOpReadVariableOp3while_lstm_cell_3_split_1_readvariableop_resource_0*
_output_shapes	
:?*
dtype0?
while/lstm_cell_3/split_1Split,while/lstm_cell_3/split_1/split_dim:output:00while/lstm_cell_3/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split?
while/lstm_cell_3/BiasAddBiasAdd"while/lstm_cell_3/MatMul:product:0"while/lstm_cell_3/split_1:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_3/BiasAdd_1BiasAdd$while/lstm_cell_3/MatMul_1:product:0"while/lstm_cell_3/split_1:output:1*
T0*(
_output_shapes
:???????????
while/lstm_cell_3/BiasAdd_2BiasAdd$while/lstm_cell_3/MatMul_2:product:0"while/lstm_cell_3/split_1:output:2*
T0*(
_output_shapes
:???????????
while/lstm_cell_3/BiasAdd_3BiasAdd$while/lstm_cell_3/MatMul_3:product:0"while/lstm_cell_3/split_1:output:3*
T0*(
_output_shapes
:???????????
 while/lstm_cell_3/ReadVariableOpReadVariableOp+while_lstm_cell_3_readvariableop_resource_0* 
_output_shapes
:
??*
dtype0v
%while/lstm_cell_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        x
'while/lstm_cell_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   x
'while/lstm_cell_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
while/lstm_cell_3/strided_sliceStridedSlice(while/lstm_cell_3/ReadVariableOp:value:0.while/lstm_cell_3/strided_slice/stack:output:00while/lstm_cell_3/strided_slice/stack_1:output:00while/lstm_cell_3/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
while/lstm_cell_3/MatMul_4MatMulwhile_placeholder_2(while/lstm_cell_3/strided_slice:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_3/addAddV2"while/lstm_cell_3/BiasAdd:output:0$while/lstm_cell_3/MatMul_4:product:0*
T0*(
_output_shapes
:??????????\
while/lstm_cell_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>^
while/lstm_cell_3/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ??
while/lstm_cell_3/MulMulwhile/lstm_cell_3/add:z:0 while/lstm_cell_3/Const:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_3/Add_1AddV2while/lstm_cell_3/Mul:z:0"while/lstm_cell_3/Const_1:output:0*
T0*(
_output_shapes
:??????????n
)while/lstm_cell_3/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
'while/lstm_cell_3/clip_by_value/MinimumMinimumwhile/lstm_cell_3/Add_1:z:02while/lstm_cell_3/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:??????????f
!while/lstm_cell_3/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
while/lstm_cell_3/clip_by_valueMaximum+while/lstm_cell_3/clip_by_value/Minimum:z:0*while/lstm_cell_3/clip_by_value/y:output:0*
T0*(
_output_shapes
:???????????
"while/lstm_cell_3/ReadVariableOp_1ReadVariableOp+while_lstm_cell_3_readvariableop_resource_0* 
_output_shapes
:
??*
dtype0x
'while/lstm_cell_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   z
)while/lstm_cell_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       z
)while/lstm_cell_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
!while/lstm_cell_3/strided_slice_1StridedSlice*while/lstm_cell_3/ReadVariableOp_1:value:00while/lstm_cell_3/strided_slice_1/stack:output:02while/lstm_cell_3/strided_slice_1/stack_1:output:02while/lstm_cell_3/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
while/lstm_cell_3/MatMul_5MatMulwhile_placeholder_2*while/lstm_cell_3/strided_slice_1:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_3/add_2AddV2$while/lstm_cell_3/BiasAdd_1:output:0$while/lstm_cell_3/MatMul_5:product:0*
T0*(
_output_shapes
:??????????^
while/lstm_cell_3/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>^
while/lstm_cell_3/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ??
while/lstm_cell_3/Mul_1Mulwhile/lstm_cell_3/add_2:z:0"while/lstm_cell_3/Const_2:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_3/Add_3AddV2while/lstm_cell_3/Mul_1:z:0"while/lstm_cell_3/Const_3:output:0*
T0*(
_output_shapes
:??????????p
+while/lstm_cell_3/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
)while/lstm_cell_3/clip_by_value_1/MinimumMinimumwhile/lstm_cell_3/Add_3:z:04while/lstm_cell_3/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:??????????h
#while/lstm_cell_3/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
!while/lstm_cell_3/clip_by_value_1Maximum-while/lstm_cell_3/clip_by_value_1/Minimum:z:0,while/lstm_cell_3/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_3/mul_2Mul%while/lstm_cell_3/clip_by_value_1:z:0while_placeholder_3*
T0*(
_output_shapes
:???????????
"while/lstm_cell_3/ReadVariableOp_2ReadVariableOp+while_lstm_cell_3_readvariableop_resource_0* 
_output_shapes
:
??*
dtype0x
'while/lstm_cell_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       z
)while/lstm_cell_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  z
)while/lstm_cell_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
!while/lstm_cell_3/strided_slice_2StridedSlice*while/lstm_cell_3/ReadVariableOp_2:value:00while/lstm_cell_3/strided_slice_2/stack:output:02while/lstm_cell_3/strided_slice_2/stack_1:output:02while/lstm_cell_3/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
while/lstm_cell_3/MatMul_6MatMulwhile_placeholder_2*while/lstm_cell_3/strided_slice_2:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_3/add_4AddV2$while/lstm_cell_3/BiasAdd_2:output:0$while/lstm_cell_3/MatMul_6:product:0*
T0*(
_output_shapes
:??????????n
while/lstm_cell_3/TanhTanhwhile/lstm_cell_3/add_4:z:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_3/mul_3Mul#while/lstm_cell_3/clip_by_value:z:0while/lstm_cell_3/Tanh:y:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_3/add_5AddV2while/lstm_cell_3/mul_2:z:0while/lstm_cell_3/mul_3:z:0*
T0*(
_output_shapes
:???????????
"while/lstm_cell_3/ReadVariableOp_3ReadVariableOp+while_lstm_cell_3_readvariableop_resource_0* 
_output_shapes
:
??*
dtype0x
'while/lstm_cell_3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  z
)while/lstm_cell_3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        z
)while/lstm_cell_3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
!while/lstm_cell_3/strided_slice_3StridedSlice*while/lstm_cell_3/ReadVariableOp_3:value:00while/lstm_cell_3/strided_slice_3/stack:output:02while/lstm_cell_3/strided_slice_3/stack_1:output:02while/lstm_cell_3/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
while/lstm_cell_3/MatMul_7MatMulwhile_placeholder_2*while/lstm_cell_3/strided_slice_3:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_3/add_6AddV2$while/lstm_cell_3/BiasAdd_3:output:0$while/lstm_cell_3/MatMul_7:product:0*
T0*(
_output_shapes
:??????????^
while/lstm_cell_3/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>^
while/lstm_cell_3/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ??
while/lstm_cell_3/Mul_4Mulwhile/lstm_cell_3/add_6:z:0"while/lstm_cell_3/Const_4:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_3/Add_7AddV2while/lstm_cell_3/Mul_4:z:0"while/lstm_cell_3/Const_5:output:0*
T0*(
_output_shapes
:??????????p
+while/lstm_cell_3/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
)while/lstm_cell_3/clip_by_value_2/MinimumMinimumwhile/lstm_cell_3/Add_7:z:04while/lstm_cell_3/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:??????????h
#while/lstm_cell_3/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
!while/lstm_cell_3/clip_by_value_2Maximum-while/lstm_cell_3/clip_by_value_2/Minimum:z:0,while/lstm_cell_3/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:??????????p
while/lstm_cell_3/Tanh_1Tanhwhile/lstm_cell_3/add_5:z:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_3/mul_5Mul%while/lstm_cell_3/clip_by_value_2:z:0while/lstm_cell_3/Tanh_1:y:0*
T0*(
_output_shapes
:???????????
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_3/mul_5:z:0*
_output_shapes
: *
element_dtype0:???M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: y
while/Identity_4Identitywhile/lstm_cell_3/mul_5:z:0^while/NoOp*
T0*(
_output_shapes
:??????????y
while/Identity_5Identitywhile/lstm_cell_3/add_5:z:0^while/NoOp*
T0*(
_output_shapes
:???????????

while/NoOpNoOp!^while/lstm_cell_3/ReadVariableOp#^while/lstm_cell_3/ReadVariableOp_1#^while/lstm_cell_3/ReadVariableOp_2#^while/lstm_cell_3/ReadVariableOp_3'^while/lstm_cell_3/split/ReadVariableOp)^while/lstm_cell_3/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"X
)while_lstm_cell_3_readvariableop_resource+while_lstm_cell_3_readvariableop_resource_0"h
1while_lstm_cell_3_split_1_readvariableop_resource3while_lstm_cell_3_split_1_readvariableop_resource_0"d
/while_lstm_cell_3_split_readvariableop_resource1while_lstm_cell_3_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2D
 while/lstm_cell_3/ReadVariableOp while/lstm_cell_3/ReadVariableOp2H
"while/lstm_cell_3/ReadVariableOp_1"while/lstm_cell_3/ReadVariableOp_12H
"while/lstm_cell_3/ReadVariableOp_2"while/lstm_cell_3/ReadVariableOp_22H
"while/lstm_cell_3/ReadVariableOp_3"while/lstm_cell_3/ReadVariableOp_32P
&while/lstm_cell_3/split/ReadVariableOp&while/lstm_cell_3/split/ReadVariableOp2T
(while/lstm_cell_3/split_1/ReadVariableOp(while/lstm_cell_3/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
??
?

G__inference_sequential_1_layer_call_and_return_conditional_losses_56749

inputsC
0lstm_2_lstm_cell_2_split_readvariableop_resource:	?A
2lstm_2_lstm_cell_2_split_1_readvariableop_resource:	?>
*lstm_2_lstm_cell_2_readvariableop_resource:
??D
0lstm_3_lstm_cell_3_split_readvariableop_resource:
??A
2lstm_3_lstm_cell_3_split_1_readvariableop_resource:	?>
*lstm_3_lstm_cell_3_readvariableop_resource:
??:
&dense_2_matmul_readvariableop_resource:
??6
'dense_2_biasadd_readvariableop_resource:	?9
&dense_3_matmul_readvariableop_resource:	?5
'dense_3_biasadd_readvariableop_resource:
identity??dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOp?dense_3/BiasAdd/ReadVariableOp?dense_3/MatMul/ReadVariableOp?!lstm_2/lstm_cell_2/ReadVariableOp?#lstm_2/lstm_cell_2/ReadVariableOp_1?#lstm_2/lstm_cell_2/ReadVariableOp_2?#lstm_2/lstm_cell_2/ReadVariableOp_3?'lstm_2/lstm_cell_2/split/ReadVariableOp?)lstm_2/lstm_cell_2/split_1/ReadVariableOp?lstm_2/while?!lstm_3/lstm_cell_3/ReadVariableOp?#lstm_3/lstm_cell_3/ReadVariableOp_1?#lstm_3/lstm_cell_3/ReadVariableOp_2?#lstm_3/lstm_cell_3/ReadVariableOp_3?'lstm_3/lstm_cell_3/split/ReadVariableOp?)lstm_3/lstm_cell_3/split_1/ReadVariableOp?lstm_3/whileB
lstm_2/ShapeShapeinputs*
T0*
_output_shapes
:d
lstm_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: f
lstm_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:f
lstm_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
lstm_2/strided_sliceStridedSlicelstm_2/Shape:output:0#lstm_2/strided_slice/stack:output:0%lstm_2/strided_slice/stack_1:output:0%lstm_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
lstm_2/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :??
lstm_2/zeros/packedPacklstm_2/strided_slice:output:0lstm_2/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:W
lstm_2/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
lstm_2/zerosFilllstm_2/zeros/packed:output:0lstm_2/zeros/Const:output:0*
T0*(
_output_shapes
:??????????Z
lstm_2/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :??
lstm_2/zeros_1/packedPacklstm_2/strided_slice:output:0 lstm_2/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:Y
lstm_2/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
lstm_2/zeros_1Filllstm_2/zeros_1/packed:output:0lstm_2/zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????j
lstm_2/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          {
lstm_2/transpose	Transposeinputslstm_2/transpose/perm:output:0*
T0*+
_output_shapes
:?????????R
lstm_2/Shape_1Shapelstm_2/transpose:y:0*
T0*
_output_shapes
:f
lstm_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: h
lstm_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
lstm_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
lstm_2/strided_slice_1StridedSlicelstm_2/Shape_1:output:0%lstm_2/strided_slice_1/stack:output:0'lstm_2/strided_slice_1/stack_1:output:0'lstm_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskm
"lstm_2/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
lstm_2/TensorArrayV2TensorListReserve+lstm_2/TensorArrayV2/element_shape:output:0lstm_2/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
<lstm_2/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
.lstm_2/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_2/transpose:y:0Elstm_2/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???f
lstm_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: h
lstm_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
lstm_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
lstm_2/strided_slice_2StridedSlicelstm_2/transpose:y:0%lstm_2/strided_slice_2/stack:output:0'lstm_2/strided_slice_2/stack_1:output:0'lstm_2/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_maskd
"lstm_2/lstm_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
'lstm_2/lstm_cell_2/split/ReadVariableOpReadVariableOp0lstm_2_lstm_cell_2_split_readvariableop_resource*
_output_shapes
:	?*
dtype0?
lstm_2/lstm_cell_2/splitSplit+lstm_2/lstm_cell_2/split/split_dim:output:0/lstm_2/lstm_cell_2/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	?:	?:	?:	?*
	num_split?
lstm_2/lstm_cell_2/MatMulMatMullstm_2/strided_slice_2:output:0!lstm_2/lstm_cell_2/split:output:0*
T0*(
_output_shapes
:???????????
lstm_2/lstm_cell_2/MatMul_1MatMullstm_2/strided_slice_2:output:0!lstm_2/lstm_cell_2/split:output:1*
T0*(
_output_shapes
:???????????
lstm_2/lstm_cell_2/MatMul_2MatMullstm_2/strided_slice_2:output:0!lstm_2/lstm_cell_2/split:output:2*
T0*(
_output_shapes
:???????????
lstm_2/lstm_cell_2/MatMul_3MatMullstm_2/strided_slice_2:output:0!lstm_2/lstm_cell_2/split:output:3*
T0*(
_output_shapes
:??????????f
$lstm_2/lstm_cell_2/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
)lstm_2/lstm_cell_2/split_1/ReadVariableOpReadVariableOp2lstm_2_lstm_cell_2_split_1_readvariableop_resource*
_output_shapes	
:?*
dtype0?
lstm_2/lstm_cell_2/split_1Split-lstm_2/lstm_cell_2/split_1/split_dim:output:01lstm_2/lstm_cell_2/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split?
lstm_2/lstm_cell_2/BiasAddBiasAdd#lstm_2/lstm_cell_2/MatMul:product:0#lstm_2/lstm_cell_2/split_1:output:0*
T0*(
_output_shapes
:???????????
lstm_2/lstm_cell_2/BiasAdd_1BiasAdd%lstm_2/lstm_cell_2/MatMul_1:product:0#lstm_2/lstm_cell_2/split_1:output:1*
T0*(
_output_shapes
:???????????
lstm_2/lstm_cell_2/BiasAdd_2BiasAdd%lstm_2/lstm_cell_2/MatMul_2:product:0#lstm_2/lstm_cell_2/split_1:output:2*
T0*(
_output_shapes
:???????????
lstm_2/lstm_cell_2/BiasAdd_3BiasAdd%lstm_2/lstm_cell_2/MatMul_3:product:0#lstm_2/lstm_cell_2/split_1:output:3*
T0*(
_output_shapes
:???????????
!lstm_2/lstm_cell_2/ReadVariableOpReadVariableOp*lstm_2_lstm_cell_2_readvariableop_resource* 
_output_shapes
:
??*
dtype0w
&lstm_2/lstm_cell_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        y
(lstm_2/lstm_cell_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       y
(lstm_2/lstm_cell_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
 lstm_2/lstm_cell_2/strided_sliceStridedSlice)lstm_2/lstm_cell_2/ReadVariableOp:value:0/lstm_2/lstm_cell_2/strided_slice/stack:output:01lstm_2/lstm_cell_2/strided_slice/stack_1:output:01lstm_2/lstm_cell_2/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
lstm_2/lstm_cell_2/MatMul_4MatMullstm_2/zeros:output:0)lstm_2/lstm_cell_2/strided_slice:output:0*
T0*(
_output_shapes
:???????????
lstm_2/lstm_cell_2/addAddV2#lstm_2/lstm_cell_2/BiasAdd:output:0%lstm_2/lstm_cell_2/MatMul_4:product:0*
T0*(
_output_shapes
:??????????]
lstm_2/lstm_cell_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>_
lstm_2/lstm_cell_2/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ??
lstm_2/lstm_cell_2/MulMullstm_2/lstm_cell_2/add:z:0!lstm_2/lstm_cell_2/Const:output:0*
T0*(
_output_shapes
:???????????
lstm_2/lstm_cell_2/Add_1AddV2lstm_2/lstm_cell_2/Mul:z:0#lstm_2/lstm_cell_2/Const_1:output:0*
T0*(
_output_shapes
:??????????o
*lstm_2/lstm_cell_2/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
(lstm_2/lstm_cell_2/clip_by_value/MinimumMinimumlstm_2/lstm_cell_2/Add_1:z:03lstm_2/lstm_cell_2/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:??????????g
"lstm_2/lstm_cell_2/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
 lstm_2/lstm_cell_2/clip_by_valueMaximum,lstm_2/lstm_cell_2/clip_by_value/Minimum:z:0+lstm_2/lstm_cell_2/clip_by_value/y:output:0*
T0*(
_output_shapes
:???????????
#lstm_2/lstm_cell_2/ReadVariableOp_1ReadVariableOp*lstm_2_lstm_cell_2_readvariableop_resource* 
_output_shapes
:
??*
dtype0y
(lstm_2/lstm_cell_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       {
*lstm_2/lstm_cell_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       {
*lstm_2/lstm_cell_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
"lstm_2/lstm_cell_2/strided_slice_1StridedSlice+lstm_2/lstm_cell_2/ReadVariableOp_1:value:01lstm_2/lstm_cell_2/strided_slice_1/stack:output:03lstm_2/lstm_cell_2/strided_slice_1/stack_1:output:03lstm_2/lstm_cell_2/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
lstm_2/lstm_cell_2/MatMul_5MatMullstm_2/zeros:output:0+lstm_2/lstm_cell_2/strided_slice_1:output:0*
T0*(
_output_shapes
:???????????
lstm_2/lstm_cell_2/add_2AddV2%lstm_2/lstm_cell_2/BiasAdd_1:output:0%lstm_2/lstm_cell_2/MatMul_5:product:0*
T0*(
_output_shapes
:??????????_
lstm_2/lstm_cell_2/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>_
lstm_2/lstm_cell_2/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ??
lstm_2/lstm_cell_2/Mul_1Mullstm_2/lstm_cell_2/add_2:z:0#lstm_2/lstm_cell_2/Const_2:output:0*
T0*(
_output_shapes
:???????????
lstm_2/lstm_cell_2/Add_3AddV2lstm_2/lstm_cell_2/Mul_1:z:0#lstm_2/lstm_cell_2/Const_3:output:0*
T0*(
_output_shapes
:??????????q
,lstm_2/lstm_cell_2/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
*lstm_2/lstm_cell_2/clip_by_value_1/MinimumMinimumlstm_2/lstm_cell_2/Add_3:z:05lstm_2/lstm_cell_2/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:??????????i
$lstm_2/lstm_cell_2/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
"lstm_2/lstm_cell_2/clip_by_value_1Maximum.lstm_2/lstm_cell_2/clip_by_value_1/Minimum:z:0-lstm_2/lstm_cell_2/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:???????????
lstm_2/lstm_cell_2/mul_2Mul&lstm_2/lstm_cell_2/clip_by_value_1:z:0lstm_2/zeros_1:output:0*
T0*(
_output_shapes
:???????????
#lstm_2/lstm_cell_2/ReadVariableOp_2ReadVariableOp*lstm_2_lstm_cell_2_readvariableop_resource* 
_output_shapes
:
??*
dtype0y
(lstm_2/lstm_cell_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       {
*lstm_2/lstm_cell_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       {
*lstm_2/lstm_cell_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
"lstm_2/lstm_cell_2/strided_slice_2StridedSlice+lstm_2/lstm_cell_2/ReadVariableOp_2:value:01lstm_2/lstm_cell_2/strided_slice_2/stack:output:03lstm_2/lstm_cell_2/strided_slice_2/stack_1:output:03lstm_2/lstm_cell_2/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
lstm_2/lstm_cell_2/MatMul_6MatMullstm_2/zeros:output:0+lstm_2/lstm_cell_2/strided_slice_2:output:0*
T0*(
_output_shapes
:???????????
lstm_2/lstm_cell_2/add_4AddV2%lstm_2/lstm_cell_2/BiasAdd_2:output:0%lstm_2/lstm_cell_2/MatMul_6:product:0*
T0*(
_output_shapes
:??????????p
lstm_2/lstm_cell_2/TanhTanhlstm_2/lstm_cell_2/add_4:z:0*
T0*(
_output_shapes
:???????????
lstm_2/lstm_cell_2/mul_3Mul$lstm_2/lstm_cell_2/clip_by_value:z:0lstm_2/lstm_cell_2/Tanh:y:0*
T0*(
_output_shapes
:???????????
lstm_2/lstm_cell_2/add_5AddV2lstm_2/lstm_cell_2/mul_2:z:0lstm_2/lstm_cell_2/mul_3:z:0*
T0*(
_output_shapes
:???????????
#lstm_2/lstm_cell_2/ReadVariableOp_3ReadVariableOp*lstm_2_lstm_cell_2_readvariableop_resource* 
_output_shapes
:
??*
dtype0y
(lstm_2/lstm_cell_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       {
*lstm_2/lstm_cell_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        {
*lstm_2/lstm_cell_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
"lstm_2/lstm_cell_2/strided_slice_3StridedSlice+lstm_2/lstm_cell_2/ReadVariableOp_3:value:01lstm_2/lstm_cell_2/strided_slice_3/stack:output:03lstm_2/lstm_cell_2/strided_slice_3/stack_1:output:03lstm_2/lstm_cell_2/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
lstm_2/lstm_cell_2/MatMul_7MatMullstm_2/zeros:output:0+lstm_2/lstm_cell_2/strided_slice_3:output:0*
T0*(
_output_shapes
:???????????
lstm_2/lstm_cell_2/add_6AddV2%lstm_2/lstm_cell_2/BiasAdd_3:output:0%lstm_2/lstm_cell_2/MatMul_7:product:0*
T0*(
_output_shapes
:??????????_
lstm_2/lstm_cell_2/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>_
lstm_2/lstm_cell_2/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ??
lstm_2/lstm_cell_2/Mul_4Mullstm_2/lstm_cell_2/add_6:z:0#lstm_2/lstm_cell_2/Const_4:output:0*
T0*(
_output_shapes
:???????????
lstm_2/lstm_cell_2/Add_7AddV2lstm_2/lstm_cell_2/Mul_4:z:0#lstm_2/lstm_cell_2/Const_5:output:0*
T0*(
_output_shapes
:??????????q
,lstm_2/lstm_cell_2/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
*lstm_2/lstm_cell_2/clip_by_value_2/MinimumMinimumlstm_2/lstm_cell_2/Add_7:z:05lstm_2/lstm_cell_2/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:??????????i
$lstm_2/lstm_cell_2/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
"lstm_2/lstm_cell_2/clip_by_value_2Maximum.lstm_2/lstm_cell_2/clip_by_value_2/Minimum:z:0-lstm_2/lstm_cell_2/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:??????????r
lstm_2/lstm_cell_2/Tanh_1Tanhlstm_2/lstm_cell_2/add_5:z:0*
T0*(
_output_shapes
:???????????
lstm_2/lstm_cell_2/mul_5Mul&lstm_2/lstm_cell_2/clip_by_value_2:z:0lstm_2/lstm_cell_2/Tanh_1:y:0*
T0*(
_output_shapes
:??????????u
$lstm_2/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
lstm_2/TensorArrayV2_1TensorListReserve-lstm_2/TensorArrayV2_1/element_shape:output:0lstm_2/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???M
lstm_2/timeConst*
_output_shapes
: *
dtype0*
value	B : j
lstm_2/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????[
lstm_2/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
lstm_2/whileWhile"lstm_2/while/loop_counter:output:0(lstm_2/while/maximum_iterations:output:0lstm_2/time:output:0lstm_2/TensorArrayV2_1:handle:0lstm_2/zeros:output:0lstm_2/zeros_1:output:0lstm_2/strided_slice_1:output:0>lstm_2/TensorArrayUnstack/TensorListFromTensor:output_handle:00lstm_2_lstm_cell_2_split_readvariableop_resource2lstm_2_lstm_cell_2_split_1_readvariableop_resource*lstm_2_lstm_cell_2_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *#
bodyR
lstm_2_while_body_56343*#
condR
lstm_2_while_cond_56342*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations ?
7lstm_2/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
)lstm_2/TensorArrayV2Stack/TensorListStackTensorListStacklstm_2/while:output:3@lstm_2/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
element_dtype0o
lstm_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????h
lstm_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: h
lstm_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
lstm_2/strided_slice_3StridedSlice2lstm_2/TensorArrayV2Stack/TensorListStack:tensor:0%lstm_2/strided_slice_3/stack:output:0'lstm_2/strided_slice_3/stack_1:output:0'lstm_2/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_maskl
lstm_2/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
lstm_2/transpose_1	Transpose2lstm_2/TensorArrayV2Stack/TensorListStack:tensor:0 lstm_2/transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????R
lstm_3/ShapeShapelstm_2/transpose_1:y:0*
T0*
_output_shapes
:d
lstm_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: f
lstm_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:f
lstm_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
lstm_3/strided_sliceStridedSlicelstm_3/Shape:output:0#lstm_3/strided_slice/stack:output:0%lstm_3/strided_slice/stack_1:output:0%lstm_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
lstm_3/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :??
lstm_3/zeros/packedPacklstm_3/strided_slice:output:0lstm_3/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:W
lstm_3/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
lstm_3/zerosFilllstm_3/zeros/packed:output:0lstm_3/zeros/Const:output:0*
T0*(
_output_shapes
:??????????Z
lstm_3/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :??
lstm_3/zeros_1/packedPacklstm_3/strided_slice:output:0 lstm_3/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:Y
lstm_3/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
lstm_3/zeros_1Filllstm_3/zeros_1/packed:output:0lstm_3/zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????j
lstm_3/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
lstm_3/transpose	Transposelstm_2/transpose_1:y:0lstm_3/transpose/perm:output:0*
T0*,
_output_shapes
:??????????R
lstm_3/Shape_1Shapelstm_3/transpose:y:0*
T0*
_output_shapes
:f
lstm_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: h
lstm_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
lstm_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
lstm_3/strided_slice_1StridedSlicelstm_3/Shape_1:output:0%lstm_3/strided_slice_1/stack:output:0'lstm_3/strided_slice_1/stack_1:output:0'lstm_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskm
"lstm_3/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
lstm_3/TensorArrayV2TensorListReserve+lstm_3/TensorArrayV2/element_shape:output:0lstm_3/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
<lstm_3/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
.lstm_3/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_3/transpose:y:0Elstm_3/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???f
lstm_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: h
lstm_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
lstm_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
lstm_3/strided_slice_2StridedSlicelstm_3/transpose:y:0%lstm_3/strided_slice_2/stack:output:0'lstm_3/strided_slice_2/stack_1:output:0'lstm_3/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_maskd
"lstm_3/lstm_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
'lstm_3/lstm_cell_3/split/ReadVariableOpReadVariableOp0lstm_3_lstm_cell_3_split_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
lstm_3/lstm_cell_3/splitSplit+lstm_3/lstm_cell_3/split/split_dim:output:0/lstm_3/lstm_cell_3/split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
??:
??:
??:
??*
	num_split?
lstm_3/lstm_cell_3/MatMulMatMullstm_3/strided_slice_2:output:0!lstm_3/lstm_cell_3/split:output:0*
T0*(
_output_shapes
:???????????
lstm_3/lstm_cell_3/MatMul_1MatMullstm_3/strided_slice_2:output:0!lstm_3/lstm_cell_3/split:output:1*
T0*(
_output_shapes
:???????????
lstm_3/lstm_cell_3/MatMul_2MatMullstm_3/strided_slice_2:output:0!lstm_3/lstm_cell_3/split:output:2*
T0*(
_output_shapes
:???????????
lstm_3/lstm_cell_3/MatMul_3MatMullstm_3/strided_slice_2:output:0!lstm_3/lstm_cell_3/split:output:3*
T0*(
_output_shapes
:??????????f
$lstm_3/lstm_cell_3/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
)lstm_3/lstm_cell_3/split_1/ReadVariableOpReadVariableOp2lstm_3_lstm_cell_3_split_1_readvariableop_resource*
_output_shapes	
:?*
dtype0?
lstm_3/lstm_cell_3/split_1Split-lstm_3/lstm_cell_3/split_1/split_dim:output:01lstm_3/lstm_cell_3/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split?
lstm_3/lstm_cell_3/BiasAddBiasAdd#lstm_3/lstm_cell_3/MatMul:product:0#lstm_3/lstm_cell_3/split_1:output:0*
T0*(
_output_shapes
:???????????
lstm_3/lstm_cell_3/BiasAdd_1BiasAdd%lstm_3/lstm_cell_3/MatMul_1:product:0#lstm_3/lstm_cell_3/split_1:output:1*
T0*(
_output_shapes
:???????????
lstm_3/lstm_cell_3/BiasAdd_2BiasAdd%lstm_3/lstm_cell_3/MatMul_2:product:0#lstm_3/lstm_cell_3/split_1:output:2*
T0*(
_output_shapes
:???????????
lstm_3/lstm_cell_3/BiasAdd_3BiasAdd%lstm_3/lstm_cell_3/MatMul_3:product:0#lstm_3/lstm_cell_3/split_1:output:3*
T0*(
_output_shapes
:???????????
!lstm_3/lstm_cell_3/ReadVariableOpReadVariableOp*lstm_3_lstm_cell_3_readvariableop_resource* 
_output_shapes
:
??*
dtype0w
&lstm_3/lstm_cell_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        y
(lstm_3/lstm_cell_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   y
(lstm_3/lstm_cell_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
 lstm_3/lstm_cell_3/strided_sliceStridedSlice)lstm_3/lstm_cell_3/ReadVariableOp:value:0/lstm_3/lstm_cell_3/strided_slice/stack:output:01lstm_3/lstm_cell_3/strided_slice/stack_1:output:01lstm_3/lstm_cell_3/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
lstm_3/lstm_cell_3/MatMul_4MatMullstm_3/zeros:output:0)lstm_3/lstm_cell_3/strided_slice:output:0*
T0*(
_output_shapes
:???????????
lstm_3/lstm_cell_3/addAddV2#lstm_3/lstm_cell_3/BiasAdd:output:0%lstm_3/lstm_cell_3/MatMul_4:product:0*
T0*(
_output_shapes
:??????????]
lstm_3/lstm_cell_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>_
lstm_3/lstm_cell_3/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ??
lstm_3/lstm_cell_3/MulMullstm_3/lstm_cell_3/add:z:0!lstm_3/lstm_cell_3/Const:output:0*
T0*(
_output_shapes
:???????????
lstm_3/lstm_cell_3/Add_1AddV2lstm_3/lstm_cell_3/Mul:z:0#lstm_3/lstm_cell_3/Const_1:output:0*
T0*(
_output_shapes
:??????????o
*lstm_3/lstm_cell_3/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
(lstm_3/lstm_cell_3/clip_by_value/MinimumMinimumlstm_3/lstm_cell_3/Add_1:z:03lstm_3/lstm_cell_3/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:??????????g
"lstm_3/lstm_cell_3/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
 lstm_3/lstm_cell_3/clip_by_valueMaximum,lstm_3/lstm_cell_3/clip_by_value/Minimum:z:0+lstm_3/lstm_cell_3/clip_by_value/y:output:0*
T0*(
_output_shapes
:???????????
#lstm_3/lstm_cell_3/ReadVariableOp_1ReadVariableOp*lstm_3_lstm_cell_3_readvariableop_resource* 
_output_shapes
:
??*
dtype0y
(lstm_3/lstm_cell_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   {
*lstm_3/lstm_cell_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       {
*lstm_3/lstm_cell_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
"lstm_3/lstm_cell_3/strided_slice_1StridedSlice+lstm_3/lstm_cell_3/ReadVariableOp_1:value:01lstm_3/lstm_cell_3/strided_slice_1/stack:output:03lstm_3/lstm_cell_3/strided_slice_1/stack_1:output:03lstm_3/lstm_cell_3/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
lstm_3/lstm_cell_3/MatMul_5MatMullstm_3/zeros:output:0+lstm_3/lstm_cell_3/strided_slice_1:output:0*
T0*(
_output_shapes
:???????????
lstm_3/lstm_cell_3/add_2AddV2%lstm_3/lstm_cell_3/BiasAdd_1:output:0%lstm_3/lstm_cell_3/MatMul_5:product:0*
T0*(
_output_shapes
:??????????_
lstm_3/lstm_cell_3/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>_
lstm_3/lstm_cell_3/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ??
lstm_3/lstm_cell_3/Mul_1Mullstm_3/lstm_cell_3/add_2:z:0#lstm_3/lstm_cell_3/Const_2:output:0*
T0*(
_output_shapes
:???????????
lstm_3/lstm_cell_3/Add_3AddV2lstm_3/lstm_cell_3/Mul_1:z:0#lstm_3/lstm_cell_3/Const_3:output:0*
T0*(
_output_shapes
:??????????q
,lstm_3/lstm_cell_3/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
*lstm_3/lstm_cell_3/clip_by_value_1/MinimumMinimumlstm_3/lstm_cell_3/Add_3:z:05lstm_3/lstm_cell_3/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:??????????i
$lstm_3/lstm_cell_3/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
"lstm_3/lstm_cell_3/clip_by_value_1Maximum.lstm_3/lstm_cell_3/clip_by_value_1/Minimum:z:0-lstm_3/lstm_cell_3/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:???????????
lstm_3/lstm_cell_3/mul_2Mul&lstm_3/lstm_cell_3/clip_by_value_1:z:0lstm_3/zeros_1:output:0*
T0*(
_output_shapes
:???????????
#lstm_3/lstm_cell_3/ReadVariableOp_2ReadVariableOp*lstm_3_lstm_cell_3_readvariableop_resource* 
_output_shapes
:
??*
dtype0y
(lstm_3/lstm_cell_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       {
*lstm_3/lstm_cell_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  {
*lstm_3/lstm_cell_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
"lstm_3/lstm_cell_3/strided_slice_2StridedSlice+lstm_3/lstm_cell_3/ReadVariableOp_2:value:01lstm_3/lstm_cell_3/strided_slice_2/stack:output:03lstm_3/lstm_cell_3/strided_slice_2/stack_1:output:03lstm_3/lstm_cell_3/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
lstm_3/lstm_cell_3/MatMul_6MatMullstm_3/zeros:output:0+lstm_3/lstm_cell_3/strided_slice_2:output:0*
T0*(
_output_shapes
:???????????
lstm_3/lstm_cell_3/add_4AddV2%lstm_3/lstm_cell_3/BiasAdd_2:output:0%lstm_3/lstm_cell_3/MatMul_6:product:0*
T0*(
_output_shapes
:??????????p
lstm_3/lstm_cell_3/TanhTanhlstm_3/lstm_cell_3/add_4:z:0*
T0*(
_output_shapes
:???????????
lstm_3/lstm_cell_3/mul_3Mul$lstm_3/lstm_cell_3/clip_by_value:z:0lstm_3/lstm_cell_3/Tanh:y:0*
T0*(
_output_shapes
:???????????
lstm_3/lstm_cell_3/add_5AddV2lstm_3/lstm_cell_3/mul_2:z:0lstm_3/lstm_cell_3/mul_3:z:0*
T0*(
_output_shapes
:???????????
#lstm_3/lstm_cell_3/ReadVariableOp_3ReadVariableOp*lstm_3_lstm_cell_3_readvariableop_resource* 
_output_shapes
:
??*
dtype0y
(lstm_3/lstm_cell_3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  {
*lstm_3/lstm_cell_3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        {
*lstm_3/lstm_cell_3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
"lstm_3/lstm_cell_3/strided_slice_3StridedSlice+lstm_3/lstm_cell_3/ReadVariableOp_3:value:01lstm_3/lstm_cell_3/strided_slice_3/stack:output:03lstm_3/lstm_cell_3/strided_slice_3/stack_1:output:03lstm_3/lstm_cell_3/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
lstm_3/lstm_cell_3/MatMul_7MatMullstm_3/zeros:output:0+lstm_3/lstm_cell_3/strided_slice_3:output:0*
T0*(
_output_shapes
:???????????
lstm_3/lstm_cell_3/add_6AddV2%lstm_3/lstm_cell_3/BiasAdd_3:output:0%lstm_3/lstm_cell_3/MatMul_7:product:0*
T0*(
_output_shapes
:??????????_
lstm_3/lstm_cell_3/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>_
lstm_3/lstm_cell_3/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ??
lstm_3/lstm_cell_3/Mul_4Mullstm_3/lstm_cell_3/add_6:z:0#lstm_3/lstm_cell_3/Const_4:output:0*
T0*(
_output_shapes
:???????????
lstm_3/lstm_cell_3/Add_7AddV2lstm_3/lstm_cell_3/Mul_4:z:0#lstm_3/lstm_cell_3/Const_5:output:0*
T0*(
_output_shapes
:??????????q
,lstm_3/lstm_cell_3/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
*lstm_3/lstm_cell_3/clip_by_value_2/MinimumMinimumlstm_3/lstm_cell_3/Add_7:z:05lstm_3/lstm_cell_3/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:??????????i
$lstm_3/lstm_cell_3/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
"lstm_3/lstm_cell_3/clip_by_value_2Maximum.lstm_3/lstm_cell_3/clip_by_value_2/Minimum:z:0-lstm_3/lstm_cell_3/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:??????????r
lstm_3/lstm_cell_3/Tanh_1Tanhlstm_3/lstm_cell_3/add_5:z:0*
T0*(
_output_shapes
:???????????
lstm_3/lstm_cell_3/mul_5Mul&lstm_3/lstm_cell_3/clip_by_value_2:z:0lstm_3/lstm_cell_3/Tanh_1:y:0*
T0*(
_output_shapes
:??????????u
$lstm_3/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   ?
lstm_3/TensorArrayV2_1TensorListReserve-lstm_3/TensorArrayV2_1/element_shape:output:0lstm_3/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???M
lstm_3/timeConst*
_output_shapes
: *
dtype0*
value	B : j
lstm_3/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????[
lstm_3/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
lstm_3/whileWhile"lstm_3/while/loop_counter:output:0(lstm_3/while/maximum_iterations:output:0lstm_3/time:output:0lstm_3/TensorArrayV2_1:handle:0lstm_3/zeros:output:0lstm_3/zeros_1:output:0lstm_3/strided_slice_1:output:0>lstm_3/TensorArrayUnstack/TensorListFromTensor:output_handle:00lstm_3_lstm_cell_3_split_readvariableop_resource2lstm_3_lstm_cell_3_split_1_readvariableop_resource*lstm_3_lstm_cell_3_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *#
bodyR
lstm_3_while_body_56595*#
condR
lstm_3_while_cond_56594*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations ?
7lstm_3/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   ?
)lstm_3/TensorArrayV2Stack/TensorListStackTensorListStacklstm_3/while:output:3@lstm_3/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
element_dtype0o
lstm_3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????h
lstm_3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: h
lstm_3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
lstm_3/strided_slice_3StridedSlice2lstm_3/TensorArrayV2Stack/TensorListStack:tensor:0%lstm_3/strided_slice_3/stack:output:0'lstm_3/strided_slice_3/stack_1:output:0'lstm_3/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_maskl
lstm_3/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
lstm_3/transpose_1	Transpose2lstm_3/TensorArrayV2Stack/TensorListStack:tensor:0 lstm_3/transpose_1/perm:output:0*
T0*,
_output_shapes
:???????????
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
dense_2/MatMulMatMullstm_3/strided_slice_3:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????a
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
dense_3/MatMulMatMuldense_2/Relu:activations:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????f
dense_3/SigmoidSigmoiddense_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????b
IdentityIdentitydense_3/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp"^lstm_2/lstm_cell_2/ReadVariableOp$^lstm_2/lstm_cell_2/ReadVariableOp_1$^lstm_2/lstm_cell_2/ReadVariableOp_2$^lstm_2/lstm_cell_2/ReadVariableOp_3(^lstm_2/lstm_cell_2/split/ReadVariableOp*^lstm_2/lstm_cell_2/split_1/ReadVariableOp^lstm_2/while"^lstm_3/lstm_cell_3/ReadVariableOp$^lstm_3/lstm_cell_3/ReadVariableOp_1$^lstm_3/lstm_cell_3/ReadVariableOp_2$^lstm_3/lstm_cell_3/ReadVariableOp_3(^lstm_3/lstm_cell_3/split/ReadVariableOp*^lstm_3/lstm_cell_3/split_1/ReadVariableOp^lstm_3/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????: : : : : : : : : : 2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2F
!lstm_2/lstm_cell_2/ReadVariableOp!lstm_2/lstm_cell_2/ReadVariableOp2J
#lstm_2/lstm_cell_2/ReadVariableOp_1#lstm_2/lstm_cell_2/ReadVariableOp_12J
#lstm_2/lstm_cell_2/ReadVariableOp_2#lstm_2/lstm_cell_2/ReadVariableOp_22J
#lstm_2/lstm_cell_2/ReadVariableOp_3#lstm_2/lstm_cell_2/ReadVariableOp_32R
'lstm_2/lstm_cell_2/split/ReadVariableOp'lstm_2/lstm_cell_2/split/ReadVariableOp2V
)lstm_2/lstm_cell_2/split_1/ReadVariableOp)lstm_2/lstm_cell_2/split_1/ReadVariableOp2
lstm_2/whilelstm_2/while2F
!lstm_3/lstm_cell_3/ReadVariableOp!lstm_3/lstm_cell_3/ReadVariableOp2J
#lstm_3/lstm_cell_3/ReadVariableOp_1#lstm_3/lstm_cell_3/ReadVariableOp_12J
#lstm_3/lstm_cell_3/ReadVariableOp_2#lstm_3/lstm_cell_3/ReadVariableOp_22J
#lstm_3/lstm_cell_3/ReadVariableOp_3#lstm_3/lstm_cell_3/ReadVariableOp_32R
'lstm_3/lstm_cell_3/split/ReadVariableOp'lstm_3/lstm_cell_3/split/ReadVariableOp2V
)lstm_3/lstm_cell_3/split_1/ReadVariableOp)lstm_3/lstm_cell_3/split_1/ReadVariableOp2
lstm_3/whilelstm_3/while:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?
lstm_3_while_body_57117*
&lstm_3_while_lstm_3_while_loop_counter0
,lstm_3_while_lstm_3_while_maximum_iterations
lstm_3_while_placeholder
lstm_3_while_placeholder_1
lstm_3_while_placeholder_2
lstm_3_while_placeholder_3)
%lstm_3_while_lstm_3_strided_slice_1_0e
alstm_3_while_tensorarrayv2read_tensorlistgetitem_lstm_3_tensorarrayunstack_tensorlistfromtensor_0L
8lstm_3_while_lstm_cell_3_split_readvariableop_resource_0:
??I
:lstm_3_while_lstm_cell_3_split_1_readvariableop_resource_0:	?F
2lstm_3_while_lstm_cell_3_readvariableop_resource_0:
??
lstm_3_while_identity
lstm_3_while_identity_1
lstm_3_while_identity_2
lstm_3_while_identity_3
lstm_3_while_identity_4
lstm_3_while_identity_5'
#lstm_3_while_lstm_3_strided_slice_1c
_lstm_3_while_tensorarrayv2read_tensorlistgetitem_lstm_3_tensorarrayunstack_tensorlistfromtensorJ
6lstm_3_while_lstm_cell_3_split_readvariableop_resource:
??G
8lstm_3_while_lstm_cell_3_split_1_readvariableop_resource:	?D
0lstm_3_while_lstm_cell_3_readvariableop_resource:
????'lstm_3/while/lstm_cell_3/ReadVariableOp?)lstm_3/while/lstm_cell_3/ReadVariableOp_1?)lstm_3/while/lstm_cell_3/ReadVariableOp_2?)lstm_3/while/lstm_cell_3/ReadVariableOp_3?-lstm_3/while/lstm_cell_3/split/ReadVariableOp?/lstm_3/while/lstm_cell_3/split_1/ReadVariableOp?
>lstm_3/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
0lstm_3/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemalstm_3_while_tensorarrayv2read_tensorlistgetitem_lstm_3_tensorarrayunstack_tensorlistfromtensor_0lstm_3_while_placeholderGlstm_3/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype0j
(lstm_3/while/lstm_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
-lstm_3/while/lstm_cell_3/split/ReadVariableOpReadVariableOp8lstm_3_while_lstm_cell_3_split_readvariableop_resource_0* 
_output_shapes
:
??*
dtype0?
lstm_3/while/lstm_cell_3/splitSplit1lstm_3/while/lstm_cell_3/split/split_dim:output:05lstm_3/while/lstm_cell_3/split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
??:
??:
??:
??*
	num_split?
lstm_3/while/lstm_cell_3/MatMulMatMul7lstm_3/while/TensorArrayV2Read/TensorListGetItem:item:0'lstm_3/while/lstm_cell_3/split:output:0*
T0*(
_output_shapes
:???????????
!lstm_3/while/lstm_cell_3/MatMul_1MatMul7lstm_3/while/TensorArrayV2Read/TensorListGetItem:item:0'lstm_3/while/lstm_cell_3/split:output:1*
T0*(
_output_shapes
:???????????
!lstm_3/while/lstm_cell_3/MatMul_2MatMul7lstm_3/while/TensorArrayV2Read/TensorListGetItem:item:0'lstm_3/while/lstm_cell_3/split:output:2*
T0*(
_output_shapes
:???????????
!lstm_3/while/lstm_cell_3/MatMul_3MatMul7lstm_3/while/TensorArrayV2Read/TensorListGetItem:item:0'lstm_3/while/lstm_cell_3/split:output:3*
T0*(
_output_shapes
:??????????l
*lstm_3/while/lstm_cell_3/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
/lstm_3/while/lstm_cell_3/split_1/ReadVariableOpReadVariableOp:lstm_3_while_lstm_cell_3_split_1_readvariableop_resource_0*
_output_shapes	
:?*
dtype0?
 lstm_3/while/lstm_cell_3/split_1Split3lstm_3/while/lstm_cell_3/split_1/split_dim:output:07lstm_3/while/lstm_cell_3/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split?
 lstm_3/while/lstm_cell_3/BiasAddBiasAdd)lstm_3/while/lstm_cell_3/MatMul:product:0)lstm_3/while/lstm_cell_3/split_1:output:0*
T0*(
_output_shapes
:???????????
"lstm_3/while/lstm_cell_3/BiasAdd_1BiasAdd+lstm_3/while/lstm_cell_3/MatMul_1:product:0)lstm_3/while/lstm_cell_3/split_1:output:1*
T0*(
_output_shapes
:???????????
"lstm_3/while/lstm_cell_3/BiasAdd_2BiasAdd+lstm_3/while/lstm_cell_3/MatMul_2:product:0)lstm_3/while/lstm_cell_3/split_1:output:2*
T0*(
_output_shapes
:???????????
"lstm_3/while/lstm_cell_3/BiasAdd_3BiasAdd+lstm_3/while/lstm_cell_3/MatMul_3:product:0)lstm_3/while/lstm_cell_3/split_1:output:3*
T0*(
_output_shapes
:???????????
'lstm_3/while/lstm_cell_3/ReadVariableOpReadVariableOp2lstm_3_while_lstm_cell_3_readvariableop_resource_0* 
_output_shapes
:
??*
dtype0}
,lstm_3/while/lstm_cell_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        
.lstm_3/while/lstm_cell_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   
.lstm_3/while/lstm_cell_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
&lstm_3/while/lstm_cell_3/strided_sliceStridedSlice/lstm_3/while/lstm_cell_3/ReadVariableOp:value:05lstm_3/while/lstm_cell_3/strided_slice/stack:output:07lstm_3/while/lstm_cell_3/strided_slice/stack_1:output:07lstm_3/while/lstm_cell_3/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
!lstm_3/while/lstm_cell_3/MatMul_4MatMullstm_3_while_placeholder_2/lstm_3/while/lstm_cell_3/strided_slice:output:0*
T0*(
_output_shapes
:???????????
lstm_3/while/lstm_cell_3/addAddV2)lstm_3/while/lstm_cell_3/BiasAdd:output:0+lstm_3/while/lstm_cell_3/MatMul_4:product:0*
T0*(
_output_shapes
:??????????c
lstm_3/while/lstm_cell_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>e
 lstm_3/while/lstm_cell_3/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ??
lstm_3/while/lstm_cell_3/MulMul lstm_3/while/lstm_cell_3/add:z:0'lstm_3/while/lstm_cell_3/Const:output:0*
T0*(
_output_shapes
:???????????
lstm_3/while/lstm_cell_3/Add_1AddV2 lstm_3/while/lstm_cell_3/Mul:z:0)lstm_3/while/lstm_cell_3/Const_1:output:0*
T0*(
_output_shapes
:??????????u
0lstm_3/while/lstm_cell_3/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
.lstm_3/while/lstm_cell_3/clip_by_value/MinimumMinimum"lstm_3/while/lstm_cell_3/Add_1:z:09lstm_3/while/lstm_cell_3/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:??????????m
(lstm_3/while/lstm_cell_3/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
&lstm_3/while/lstm_cell_3/clip_by_valueMaximum2lstm_3/while/lstm_cell_3/clip_by_value/Minimum:z:01lstm_3/while/lstm_cell_3/clip_by_value/y:output:0*
T0*(
_output_shapes
:???????????
)lstm_3/while/lstm_cell_3/ReadVariableOp_1ReadVariableOp2lstm_3_while_lstm_cell_3_readvariableop_resource_0* 
_output_shapes
:
??*
dtype0
.lstm_3/while/lstm_cell_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   ?
0lstm_3/while/lstm_cell_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
0lstm_3/while/lstm_cell_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
(lstm_3/while/lstm_cell_3/strided_slice_1StridedSlice1lstm_3/while/lstm_cell_3/ReadVariableOp_1:value:07lstm_3/while/lstm_cell_3/strided_slice_1/stack:output:09lstm_3/while/lstm_cell_3/strided_slice_1/stack_1:output:09lstm_3/while/lstm_cell_3/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
!lstm_3/while/lstm_cell_3/MatMul_5MatMullstm_3_while_placeholder_21lstm_3/while/lstm_cell_3/strided_slice_1:output:0*
T0*(
_output_shapes
:???????????
lstm_3/while/lstm_cell_3/add_2AddV2+lstm_3/while/lstm_cell_3/BiasAdd_1:output:0+lstm_3/while/lstm_cell_3/MatMul_5:product:0*
T0*(
_output_shapes
:??????????e
 lstm_3/while/lstm_cell_3/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>e
 lstm_3/while/lstm_cell_3/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ??
lstm_3/while/lstm_cell_3/Mul_1Mul"lstm_3/while/lstm_cell_3/add_2:z:0)lstm_3/while/lstm_cell_3/Const_2:output:0*
T0*(
_output_shapes
:???????????
lstm_3/while/lstm_cell_3/Add_3AddV2"lstm_3/while/lstm_cell_3/Mul_1:z:0)lstm_3/while/lstm_cell_3/Const_3:output:0*
T0*(
_output_shapes
:??????????w
2lstm_3/while/lstm_cell_3/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
0lstm_3/while/lstm_cell_3/clip_by_value_1/MinimumMinimum"lstm_3/while/lstm_cell_3/Add_3:z:0;lstm_3/while/lstm_cell_3/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:??????????o
*lstm_3/while/lstm_cell_3/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
(lstm_3/while/lstm_cell_3/clip_by_value_1Maximum4lstm_3/while/lstm_cell_3/clip_by_value_1/Minimum:z:03lstm_3/while/lstm_cell_3/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:???????????
lstm_3/while/lstm_cell_3/mul_2Mul,lstm_3/while/lstm_cell_3/clip_by_value_1:z:0lstm_3_while_placeholder_3*
T0*(
_output_shapes
:???????????
)lstm_3/while/lstm_cell_3/ReadVariableOp_2ReadVariableOp2lstm_3_while_lstm_cell_3_readvariableop_resource_0* 
_output_shapes
:
??*
dtype0
.lstm_3/while/lstm_cell_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       ?
0lstm_3/while/lstm_cell_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  ?
0lstm_3/while/lstm_cell_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
(lstm_3/while/lstm_cell_3/strided_slice_2StridedSlice1lstm_3/while/lstm_cell_3/ReadVariableOp_2:value:07lstm_3/while/lstm_cell_3/strided_slice_2/stack:output:09lstm_3/while/lstm_cell_3/strided_slice_2/stack_1:output:09lstm_3/while/lstm_cell_3/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
!lstm_3/while/lstm_cell_3/MatMul_6MatMullstm_3_while_placeholder_21lstm_3/while/lstm_cell_3/strided_slice_2:output:0*
T0*(
_output_shapes
:???????????
lstm_3/while/lstm_cell_3/add_4AddV2+lstm_3/while/lstm_cell_3/BiasAdd_2:output:0+lstm_3/while/lstm_cell_3/MatMul_6:product:0*
T0*(
_output_shapes
:??????????|
lstm_3/while/lstm_cell_3/TanhTanh"lstm_3/while/lstm_cell_3/add_4:z:0*
T0*(
_output_shapes
:???????????
lstm_3/while/lstm_cell_3/mul_3Mul*lstm_3/while/lstm_cell_3/clip_by_value:z:0!lstm_3/while/lstm_cell_3/Tanh:y:0*
T0*(
_output_shapes
:???????????
lstm_3/while/lstm_cell_3/add_5AddV2"lstm_3/while/lstm_cell_3/mul_2:z:0"lstm_3/while/lstm_cell_3/mul_3:z:0*
T0*(
_output_shapes
:???????????
)lstm_3/while/lstm_cell_3/ReadVariableOp_3ReadVariableOp2lstm_3_while_lstm_cell_3_readvariableop_resource_0* 
_output_shapes
:
??*
dtype0
.lstm_3/while/lstm_cell_3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  ?
0lstm_3/while/lstm_cell_3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        ?
0lstm_3/while/lstm_cell_3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
(lstm_3/while/lstm_cell_3/strided_slice_3StridedSlice1lstm_3/while/lstm_cell_3/ReadVariableOp_3:value:07lstm_3/while/lstm_cell_3/strided_slice_3/stack:output:09lstm_3/while/lstm_cell_3/strided_slice_3/stack_1:output:09lstm_3/while/lstm_cell_3/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
!lstm_3/while/lstm_cell_3/MatMul_7MatMullstm_3_while_placeholder_21lstm_3/while/lstm_cell_3/strided_slice_3:output:0*
T0*(
_output_shapes
:???????????
lstm_3/while/lstm_cell_3/add_6AddV2+lstm_3/while/lstm_cell_3/BiasAdd_3:output:0+lstm_3/while/lstm_cell_3/MatMul_7:product:0*
T0*(
_output_shapes
:??????????e
 lstm_3/while/lstm_cell_3/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>e
 lstm_3/while/lstm_cell_3/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ??
lstm_3/while/lstm_cell_3/Mul_4Mul"lstm_3/while/lstm_cell_3/add_6:z:0)lstm_3/while/lstm_cell_3/Const_4:output:0*
T0*(
_output_shapes
:???????????
lstm_3/while/lstm_cell_3/Add_7AddV2"lstm_3/while/lstm_cell_3/Mul_4:z:0)lstm_3/while/lstm_cell_3/Const_5:output:0*
T0*(
_output_shapes
:??????????w
2lstm_3/while/lstm_cell_3/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
0lstm_3/while/lstm_cell_3/clip_by_value_2/MinimumMinimum"lstm_3/while/lstm_cell_3/Add_7:z:0;lstm_3/while/lstm_cell_3/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:??????????o
*lstm_3/while/lstm_cell_3/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
(lstm_3/while/lstm_cell_3/clip_by_value_2Maximum4lstm_3/while/lstm_cell_3/clip_by_value_2/Minimum:z:03lstm_3/while/lstm_cell_3/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:??????????~
lstm_3/while/lstm_cell_3/Tanh_1Tanh"lstm_3/while/lstm_cell_3/add_5:z:0*
T0*(
_output_shapes
:???????????
lstm_3/while/lstm_cell_3/mul_5Mul,lstm_3/while/lstm_cell_3/clip_by_value_2:z:0#lstm_3/while/lstm_cell_3/Tanh_1:y:0*
T0*(
_output_shapes
:???????????
1lstm_3/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_3_while_placeholder_1lstm_3_while_placeholder"lstm_3/while/lstm_cell_3/mul_5:z:0*
_output_shapes
: *
element_dtype0:???T
lstm_3/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :q
lstm_3/while/addAddV2lstm_3_while_placeholderlstm_3/while/add/y:output:0*
T0*
_output_shapes
: V
lstm_3/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :?
lstm_3/while/add_1AddV2&lstm_3_while_lstm_3_while_loop_counterlstm_3/while/add_1/y:output:0*
T0*
_output_shapes
: n
lstm_3/while/IdentityIdentitylstm_3/while/add_1:z:0^lstm_3/while/NoOp*
T0*
_output_shapes
: ?
lstm_3/while/Identity_1Identity,lstm_3_while_lstm_3_while_maximum_iterations^lstm_3/while/NoOp*
T0*
_output_shapes
: n
lstm_3/while/Identity_2Identitylstm_3/while/add:z:0^lstm_3/while/NoOp*
T0*
_output_shapes
: ?
lstm_3/while/Identity_3IdentityAlstm_3/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_3/while/NoOp*
T0*
_output_shapes
: ?
lstm_3/while/Identity_4Identity"lstm_3/while/lstm_cell_3/mul_5:z:0^lstm_3/while/NoOp*
T0*(
_output_shapes
:???????????
lstm_3/while/Identity_5Identity"lstm_3/while/lstm_cell_3/add_5:z:0^lstm_3/while/NoOp*
T0*(
_output_shapes
:???????????
lstm_3/while/NoOpNoOp(^lstm_3/while/lstm_cell_3/ReadVariableOp*^lstm_3/while/lstm_cell_3/ReadVariableOp_1*^lstm_3/while/lstm_cell_3/ReadVariableOp_2*^lstm_3/while/lstm_cell_3/ReadVariableOp_3.^lstm_3/while/lstm_cell_3/split/ReadVariableOp0^lstm_3/while/lstm_cell_3/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "7
lstm_3_while_identitylstm_3/while/Identity:output:0";
lstm_3_while_identity_1 lstm_3/while/Identity_1:output:0";
lstm_3_while_identity_2 lstm_3/while/Identity_2:output:0";
lstm_3_while_identity_3 lstm_3/while/Identity_3:output:0";
lstm_3_while_identity_4 lstm_3/while/Identity_4:output:0";
lstm_3_while_identity_5 lstm_3/while/Identity_5:output:0"L
#lstm_3_while_lstm_3_strided_slice_1%lstm_3_while_lstm_3_strided_slice_1_0"f
0lstm_3_while_lstm_cell_3_readvariableop_resource2lstm_3_while_lstm_cell_3_readvariableop_resource_0"v
8lstm_3_while_lstm_cell_3_split_1_readvariableop_resource:lstm_3_while_lstm_cell_3_split_1_readvariableop_resource_0"r
6lstm_3_while_lstm_cell_3_split_readvariableop_resource8lstm_3_while_lstm_cell_3_split_readvariableop_resource_0"?
_lstm_3_while_tensorarrayv2read_tensorlistgetitem_lstm_3_tensorarrayunstack_tensorlistfromtensoralstm_3_while_tensorarrayv2read_tensorlistgetitem_lstm_3_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2R
'lstm_3/while/lstm_cell_3/ReadVariableOp'lstm_3/while/lstm_cell_3/ReadVariableOp2V
)lstm_3/while/lstm_cell_3/ReadVariableOp_1)lstm_3/while/lstm_cell_3/ReadVariableOp_12V
)lstm_3/while/lstm_cell_3/ReadVariableOp_2)lstm_3/while/lstm_cell_3/ReadVariableOp_22V
)lstm_3/while/lstm_cell_3/ReadVariableOp_3)lstm_3/while/lstm_cell_3/ReadVariableOp_32^
-lstm_3/while/lstm_cell_3/split/ReadVariableOp-lstm_3/while/lstm_cell_3/split/ReadVariableOp2b
/lstm_3/while/lstm_cell_3/split_1/ReadVariableOp/lstm_3/while/lstm_cell_3/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?{
?	
while_body_55608
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
1while_lstm_cell_3_split_readvariableop_resource_0:
??B
3while_lstm_cell_3_split_1_readvariableop_resource_0:	??
+while_lstm_cell_3_readvariableop_resource_0:
??
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
/while_lstm_cell_3_split_readvariableop_resource:
??@
1while_lstm_cell_3_split_1_readvariableop_resource:	?=
)while_lstm_cell_3_readvariableop_resource:
???? while/lstm_cell_3/ReadVariableOp?"while/lstm_cell_3/ReadVariableOp_1?"while/lstm_cell_3/ReadVariableOp_2?"while/lstm_cell_3/ReadVariableOp_3?&while/lstm_cell_3/split/ReadVariableOp?(while/lstm_cell_3/split_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype0c
!while/lstm_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
&while/lstm_cell_3/split/ReadVariableOpReadVariableOp1while_lstm_cell_3_split_readvariableop_resource_0* 
_output_shapes
:
??*
dtype0?
while/lstm_cell_3/splitSplit*while/lstm_cell_3/split/split_dim:output:0.while/lstm_cell_3/split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
??:
??:
??:
??*
	num_split?
while/lstm_cell_3/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_3/split:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_3/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_3/split:output:1*
T0*(
_output_shapes
:???????????
while/lstm_cell_3/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_3/split:output:2*
T0*(
_output_shapes
:???????????
while/lstm_cell_3/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_3/split:output:3*
T0*(
_output_shapes
:??????????e
#while/lstm_cell_3/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
(while/lstm_cell_3/split_1/ReadVariableOpReadVariableOp3while_lstm_cell_3_split_1_readvariableop_resource_0*
_output_shapes	
:?*
dtype0?
while/lstm_cell_3/split_1Split,while/lstm_cell_3/split_1/split_dim:output:00while/lstm_cell_3/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split?
while/lstm_cell_3/BiasAddBiasAdd"while/lstm_cell_3/MatMul:product:0"while/lstm_cell_3/split_1:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_3/BiasAdd_1BiasAdd$while/lstm_cell_3/MatMul_1:product:0"while/lstm_cell_3/split_1:output:1*
T0*(
_output_shapes
:???????????
while/lstm_cell_3/BiasAdd_2BiasAdd$while/lstm_cell_3/MatMul_2:product:0"while/lstm_cell_3/split_1:output:2*
T0*(
_output_shapes
:???????????
while/lstm_cell_3/BiasAdd_3BiasAdd$while/lstm_cell_3/MatMul_3:product:0"while/lstm_cell_3/split_1:output:3*
T0*(
_output_shapes
:???????????
 while/lstm_cell_3/ReadVariableOpReadVariableOp+while_lstm_cell_3_readvariableop_resource_0* 
_output_shapes
:
??*
dtype0v
%while/lstm_cell_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        x
'while/lstm_cell_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   x
'while/lstm_cell_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
while/lstm_cell_3/strided_sliceStridedSlice(while/lstm_cell_3/ReadVariableOp:value:0.while/lstm_cell_3/strided_slice/stack:output:00while/lstm_cell_3/strided_slice/stack_1:output:00while/lstm_cell_3/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
while/lstm_cell_3/MatMul_4MatMulwhile_placeholder_2(while/lstm_cell_3/strided_slice:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_3/addAddV2"while/lstm_cell_3/BiasAdd:output:0$while/lstm_cell_3/MatMul_4:product:0*
T0*(
_output_shapes
:??????????\
while/lstm_cell_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>^
while/lstm_cell_3/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ??
while/lstm_cell_3/MulMulwhile/lstm_cell_3/add:z:0 while/lstm_cell_3/Const:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_3/Add_1AddV2while/lstm_cell_3/Mul:z:0"while/lstm_cell_3/Const_1:output:0*
T0*(
_output_shapes
:??????????n
)while/lstm_cell_3/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
'while/lstm_cell_3/clip_by_value/MinimumMinimumwhile/lstm_cell_3/Add_1:z:02while/lstm_cell_3/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:??????????f
!while/lstm_cell_3/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
while/lstm_cell_3/clip_by_valueMaximum+while/lstm_cell_3/clip_by_value/Minimum:z:0*while/lstm_cell_3/clip_by_value/y:output:0*
T0*(
_output_shapes
:???????????
"while/lstm_cell_3/ReadVariableOp_1ReadVariableOp+while_lstm_cell_3_readvariableop_resource_0* 
_output_shapes
:
??*
dtype0x
'while/lstm_cell_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   z
)while/lstm_cell_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       z
)while/lstm_cell_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
!while/lstm_cell_3/strided_slice_1StridedSlice*while/lstm_cell_3/ReadVariableOp_1:value:00while/lstm_cell_3/strided_slice_1/stack:output:02while/lstm_cell_3/strided_slice_1/stack_1:output:02while/lstm_cell_3/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
while/lstm_cell_3/MatMul_5MatMulwhile_placeholder_2*while/lstm_cell_3/strided_slice_1:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_3/add_2AddV2$while/lstm_cell_3/BiasAdd_1:output:0$while/lstm_cell_3/MatMul_5:product:0*
T0*(
_output_shapes
:??????????^
while/lstm_cell_3/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>^
while/lstm_cell_3/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ??
while/lstm_cell_3/Mul_1Mulwhile/lstm_cell_3/add_2:z:0"while/lstm_cell_3/Const_2:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_3/Add_3AddV2while/lstm_cell_3/Mul_1:z:0"while/lstm_cell_3/Const_3:output:0*
T0*(
_output_shapes
:??????????p
+while/lstm_cell_3/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
)while/lstm_cell_3/clip_by_value_1/MinimumMinimumwhile/lstm_cell_3/Add_3:z:04while/lstm_cell_3/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:??????????h
#while/lstm_cell_3/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
!while/lstm_cell_3/clip_by_value_1Maximum-while/lstm_cell_3/clip_by_value_1/Minimum:z:0,while/lstm_cell_3/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_3/mul_2Mul%while/lstm_cell_3/clip_by_value_1:z:0while_placeholder_3*
T0*(
_output_shapes
:???????????
"while/lstm_cell_3/ReadVariableOp_2ReadVariableOp+while_lstm_cell_3_readvariableop_resource_0* 
_output_shapes
:
??*
dtype0x
'while/lstm_cell_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       z
)while/lstm_cell_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  z
)while/lstm_cell_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
!while/lstm_cell_3/strided_slice_2StridedSlice*while/lstm_cell_3/ReadVariableOp_2:value:00while/lstm_cell_3/strided_slice_2/stack:output:02while/lstm_cell_3/strided_slice_2/stack_1:output:02while/lstm_cell_3/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
while/lstm_cell_3/MatMul_6MatMulwhile_placeholder_2*while/lstm_cell_3/strided_slice_2:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_3/add_4AddV2$while/lstm_cell_3/BiasAdd_2:output:0$while/lstm_cell_3/MatMul_6:product:0*
T0*(
_output_shapes
:??????????n
while/lstm_cell_3/TanhTanhwhile/lstm_cell_3/add_4:z:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_3/mul_3Mul#while/lstm_cell_3/clip_by_value:z:0while/lstm_cell_3/Tanh:y:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_3/add_5AddV2while/lstm_cell_3/mul_2:z:0while/lstm_cell_3/mul_3:z:0*
T0*(
_output_shapes
:???????????
"while/lstm_cell_3/ReadVariableOp_3ReadVariableOp+while_lstm_cell_3_readvariableop_resource_0* 
_output_shapes
:
??*
dtype0x
'while/lstm_cell_3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  z
)while/lstm_cell_3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        z
)while/lstm_cell_3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
!while/lstm_cell_3/strided_slice_3StridedSlice*while/lstm_cell_3/ReadVariableOp_3:value:00while/lstm_cell_3/strided_slice_3/stack:output:02while/lstm_cell_3/strided_slice_3/stack_1:output:02while/lstm_cell_3/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
while/lstm_cell_3/MatMul_7MatMulwhile_placeholder_2*while/lstm_cell_3/strided_slice_3:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_3/add_6AddV2$while/lstm_cell_3/BiasAdd_3:output:0$while/lstm_cell_3/MatMul_7:product:0*
T0*(
_output_shapes
:??????????^
while/lstm_cell_3/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>^
while/lstm_cell_3/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ??
while/lstm_cell_3/Mul_4Mulwhile/lstm_cell_3/add_6:z:0"while/lstm_cell_3/Const_4:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_3/Add_7AddV2while/lstm_cell_3/Mul_4:z:0"while/lstm_cell_3/Const_5:output:0*
T0*(
_output_shapes
:??????????p
+while/lstm_cell_3/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
)while/lstm_cell_3/clip_by_value_2/MinimumMinimumwhile/lstm_cell_3/Add_7:z:04while/lstm_cell_3/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:??????????h
#while/lstm_cell_3/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
!while/lstm_cell_3/clip_by_value_2Maximum-while/lstm_cell_3/clip_by_value_2/Minimum:z:0,while/lstm_cell_3/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:??????????p
while/lstm_cell_3/Tanh_1Tanhwhile/lstm_cell_3/add_5:z:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_3/mul_5Mul%while/lstm_cell_3/clip_by_value_2:z:0while/lstm_cell_3/Tanh_1:y:0*
T0*(
_output_shapes
:???????????
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_3/mul_5:z:0*
_output_shapes
: *
element_dtype0:???M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: y
while/Identity_4Identitywhile/lstm_cell_3/mul_5:z:0^while/NoOp*
T0*(
_output_shapes
:??????????y
while/Identity_5Identitywhile/lstm_cell_3/add_5:z:0^while/NoOp*
T0*(
_output_shapes
:???????????

while/NoOpNoOp!^while/lstm_cell_3/ReadVariableOp#^while/lstm_cell_3/ReadVariableOp_1#^while/lstm_cell_3/ReadVariableOp_2#^while/lstm_cell_3/ReadVariableOp_3'^while/lstm_cell_3/split/ReadVariableOp)^while/lstm_cell_3/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"X
)while_lstm_cell_3_readvariableop_resource+while_lstm_cell_3_readvariableop_resource_0"h
1while_lstm_cell_3_split_1_readvariableop_resource3while_lstm_cell_3_split_1_readvariableop_resource_0"d
/while_lstm_cell_3_split_readvariableop_resource1while_lstm_cell_3_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2D
 while/lstm_cell_3/ReadVariableOp while/lstm_cell_3/ReadVariableOp2H
"while/lstm_cell_3/ReadVariableOp_1"while/lstm_cell_3/ReadVariableOp_12H
"while/lstm_cell_3/ReadVariableOp_2"while/lstm_cell_3/ReadVariableOp_22H
"while/lstm_cell_3/ReadVariableOp_3"while/lstm_cell_3/ReadVariableOp_32P
&while/lstm_cell_3/split/ReadVariableOp&while/lstm_cell_3/split/ReadVariableOp2T
(while/lstm_cell_3/split_1/ReadVariableOp(while/lstm_cell_3/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
??
?
A__inference_lstm_2_layer_call_and_return_conditional_losses_55130

inputs<
)lstm_cell_2_split_readvariableop_resource:	?:
+lstm_cell_2_split_1_readvariableop_resource:	?7
#lstm_cell_2_readvariableop_resource:
??
identity??lstm_cell_2/ReadVariableOp?lstm_cell_2/ReadVariableOp_1?lstm_cell_2/ReadVariableOp_2?lstm_cell_2/ReadVariableOp_3? lstm_cell_2/split/ReadVariableOp?"lstm_cell_2/split_1/ReadVariableOp?while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????S
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:?????????D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask]
lstm_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
 lstm_cell_2/split/ReadVariableOpReadVariableOp)lstm_cell_2_split_readvariableop_resource*
_output_shapes
:	?*
dtype0?
lstm_cell_2/splitSplit$lstm_cell_2/split/split_dim:output:0(lstm_cell_2/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	?:	?:	?:	?*
	num_split?
lstm_cell_2/MatMulMatMulstrided_slice_2:output:0lstm_cell_2/split:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_2/MatMul_1MatMulstrided_slice_2:output:0lstm_cell_2/split:output:1*
T0*(
_output_shapes
:???????????
lstm_cell_2/MatMul_2MatMulstrided_slice_2:output:0lstm_cell_2/split:output:2*
T0*(
_output_shapes
:???????????
lstm_cell_2/MatMul_3MatMulstrided_slice_2:output:0lstm_cell_2/split:output:3*
T0*(
_output_shapes
:??????????_
lstm_cell_2/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
"lstm_cell_2/split_1/ReadVariableOpReadVariableOp+lstm_cell_2_split_1_readvariableop_resource*
_output_shapes	
:?*
dtype0?
lstm_cell_2/split_1Split&lstm_cell_2/split_1/split_dim:output:0*lstm_cell_2/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split?
lstm_cell_2/BiasAddBiasAddlstm_cell_2/MatMul:product:0lstm_cell_2/split_1:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_2/BiasAdd_1BiasAddlstm_cell_2/MatMul_1:product:0lstm_cell_2/split_1:output:1*
T0*(
_output_shapes
:???????????
lstm_cell_2/BiasAdd_2BiasAddlstm_cell_2/MatMul_2:product:0lstm_cell_2/split_1:output:2*
T0*(
_output_shapes
:???????????
lstm_cell_2/BiasAdd_3BiasAddlstm_cell_2/MatMul_3:product:0lstm_cell_2/split_1:output:3*
T0*(
_output_shapes
:???????????
lstm_cell_2/ReadVariableOpReadVariableOp#lstm_cell_2_readvariableop_resource* 
_output_shapes
:
??*
dtype0p
lstm_cell_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        r
!lstm_cell_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       r
!lstm_cell_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
lstm_cell_2/strided_sliceStridedSlice"lstm_cell_2/ReadVariableOp:value:0(lstm_cell_2/strided_slice/stack:output:0*lstm_cell_2/strided_slice/stack_1:output:0*lstm_cell_2/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
lstm_cell_2/MatMul_4MatMulzeros:output:0"lstm_cell_2/strided_slice:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_2/addAddV2lstm_cell_2/BiasAdd:output:0lstm_cell_2/MatMul_4:product:0*
T0*(
_output_shapes
:??????????V
lstm_cell_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>X
lstm_cell_2/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?z
lstm_cell_2/MulMullstm_cell_2/add:z:0lstm_cell_2/Const:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_2/Add_1AddV2lstm_cell_2/Mul:z:0lstm_cell_2/Const_1:output:0*
T0*(
_output_shapes
:??????????h
#lstm_cell_2/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
!lstm_cell_2/clip_by_value/MinimumMinimumlstm_cell_2/Add_1:z:0,lstm_cell_2/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:??????????`
lstm_cell_2/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
lstm_cell_2/clip_by_valueMaximum%lstm_cell_2/clip_by_value/Minimum:z:0$lstm_cell_2/clip_by_value/y:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_2/ReadVariableOp_1ReadVariableOp#lstm_cell_2_readvariableop_resource* 
_output_shapes
:
??*
dtype0r
!lstm_cell_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       t
#lstm_cell_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       t
#lstm_cell_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
lstm_cell_2/strided_slice_1StridedSlice$lstm_cell_2/ReadVariableOp_1:value:0*lstm_cell_2/strided_slice_1/stack:output:0,lstm_cell_2/strided_slice_1/stack_1:output:0,lstm_cell_2/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
lstm_cell_2/MatMul_5MatMulzeros:output:0$lstm_cell_2/strided_slice_1:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_2/add_2AddV2lstm_cell_2/BiasAdd_1:output:0lstm_cell_2/MatMul_5:product:0*
T0*(
_output_shapes
:??????????X
lstm_cell_2/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>X
lstm_cell_2/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ??
lstm_cell_2/Mul_1Mullstm_cell_2/add_2:z:0lstm_cell_2/Const_2:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_2/Add_3AddV2lstm_cell_2/Mul_1:z:0lstm_cell_2/Const_3:output:0*
T0*(
_output_shapes
:??????????j
%lstm_cell_2/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
#lstm_cell_2/clip_by_value_1/MinimumMinimumlstm_cell_2/Add_3:z:0.lstm_cell_2/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:??????????b
lstm_cell_2/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
lstm_cell_2/clip_by_value_1Maximum'lstm_cell_2/clip_by_value_1/Minimum:z:0&lstm_cell_2/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:??????????~
lstm_cell_2/mul_2Mullstm_cell_2/clip_by_value_1:z:0zeros_1:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_2/ReadVariableOp_2ReadVariableOp#lstm_cell_2_readvariableop_resource* 
_output_shapes
:
??*
dtype0r
!lstm_cell_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       t
#lstm_cell_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       t
#lstm_cell_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
lstm_cell_2/strided_slice_2StridedSlice$lstm_cell_2/ReadVariableOp_2:value:0*lstm_cell_2/strided_slice_2/stack:output:0,lstm_cell_2/strided_slice_2/stack_1:output:0,lstm_cell_2/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
lstm_cell_2/MatMul_6MatMulzeros:output:0$lstm_cell_2/strided_slice_2:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_2/add_4AddV2lstm_cell_2/BiasAdd_2:output:0lstm_cell_2/MatMul_6:product:0*
T0*(
_output_shapes
:??????????b
lstm_cell_2/TanhTanhlstm_cell_2/add_4:z:0*
T0*(
_output_shapes
:???????????
lstm_cell_2/mul_3Mullstm_cell_2/clip_by_value:z:0lstm_cell_2/Tanh:y:0*
T0*(
_output_shapes
:??????????{
lstm_cell_2/add_5AddV2lstm_cell_2/mul_2:z:0lstm_cell_2/mul_3:z:0*
T0*(
_output_shapes
:???????????
lstm_cell_2/ReadVariableOp_3ReadVariableOp#lstm_cell_2_readvariableop_resource* 
_output_shapes
:
??*
dtype0r
!lstm_cell_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       t
#lstm_cell_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        t
#lstm_cell_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
lstm_cell_2/strided_slice_3StridedSlice$lstm_cell_2/ReadVariableOp_3:value:0*lstm_cell_2/strided_slice_3/stack:output:0,lstm_cell_2/strided_slice_3/stack_1:output:0,lstm_cell_2/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
lstm_cell_2/MatMul_7MatMulzeros:output:0$lstm_cell_2/strided_slice_3:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_2/add_6AddV2lstm_cell_2/BiasAdd_3:output:0lstm_cell_2/MatMul_7:product:0*
T0*(
_output_shapes
:??????????X
lstm_cell_2/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>X
lstm_cell_2/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ??
lstm_cell_2/Mul_4Mullstm_cell_2/add_6:z:0lstm_cell_2/Const_4:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_2/Add_7AddV2lstm_cell_2/Mul_4:z:0lstm_cell_2/Const_5:output:0*
T0*(
_output_shapes
:??????????j
%lstm_cell_2/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
#lstm_cell_2/clip_by_value_2/MinimumMinimumlstm_cell_2/Add_7:z:0.lstm_cell_2/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:??????????b
lstm_cell_2/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
lstm_cell_2/clip_by_value_2Maximum'lstm_cell_2/clip_by_value_2/Minimum:z:0&lstm_cell_2/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:??????????d
lstm_cell_2/Tanh_1Tanhlstm_cell_2/add_5:z:0*
T0*(
_output_shapes
:???????????
lstm_cell_2/mul_5Mullstm_cell_2/clip_by_value_2:z:0lstm_cell_2/Tanh_1:y:0*
T0*(
_output_shapes
:??????????n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)lstm_cell_2_split_readvariableop_resource+lstm_cell_2_split_1_readvariableop_resource#lstm_cell_2_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_54990*
condR
while_cond_54989*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????c
IdentityIdentitytranspose_1:y:0^NoOp*
T0*,
_output_shapes
:???????????
NoOpNoOp^lstm_cell_2/ReadVariableOp^lstm_cell_2/ReadVariableOp_1^lstm_cell_2/ReadVariableOp_2^lstm_cell_2/ReadVariableOp_3!^lstm_cell_2/split/ReadVariableOp#^lstm_cell_2/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????: : : 28
lstm_cell_2/ReadVariableOplstm_cell_2/ReadVariableOp2<
lstm_cell_2/ReadVariableOp_1lstm_cell_2/ReadVariableOp_12<
lstm_cell_2/ReadVariableOp_2lstm_cell_2/ReadVariableOp_22<
lstm_cell_2/ReadVariableOp_3lstm_cell_2/ReadVariableOp_32D
 lstm_cell_2/split/ReadVariableOp lstm_cell_2/split/ReadVariableOp2H
"lstm_cell_2/split_1/ReadVariableOp"lstm_cell_2/split_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?{
?	
while_body_58205
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_lstm_cell_2_split_readvariableop_resource_0:	?B
3while_lstm_cell_2_split_1_readvariableop_resource_0:	??
+while_lstm_cell_2_readvariableop_resource_0:
??
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_lstm_cell_2_split_readvariableop_resource:	?@
1while_lstm_cell_2_split_1_readvariableop_resource:	?=
)while_lstm_cell_2_readvariableop_resource:
???? while/lstm_cell_2/ReadVariableOp?"while/lstm_cell_2/ReadVariableOp_1?"while/lstm_cell_2/ReadVariableOp_2?"while/lstm_cell_2/ReadVariableOp_3?&while/lstm_cell_2/split/ReadVariableOp?(while/lstm_cell_2/split_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0c
!while/lstm_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
&while/lstm_cell_2/split/ReadVariableOpReadVariableOp1while_lstm_cell_2_split_readvariableop_resource_0*
_output_shapes
:	?*
dtype0?
while/lstm_cell_2/splitSplit*while/lstm_cell_2/split/split_dim:output:0.while/lstm_cell_2/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	?:	?:	?:	?*
	num_split?
while/lstm_cell_2/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_2/split:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_2/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_2/split:output:1*
T0*(
_output_shapes
:???????????
while/lstm_cell_2/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_2/split:output:2*
T0*(
_output_shapes
:???????????
while/lstm_cell_2/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_2/split:output:3*
T0*(
_output_shapes
:??????????e
#while/lstm_cell_2/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
(while/lstm_cell_2/split_1/ReadVariableOpReadVariableOp3while_lstm_cell_2_split_1_readvariableop_resource_0*
_output_shapes	
:?*
dtype0?
while/lstm_cell_2/split_1Split,while/lstm_cell_2/split_1/split_dim:output:00while/lstm_cell_2/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split?
while/lstm_cell_2/BiasAddBiasAdd"while/lstm_cell_2/MatMul:product:0"while/lstm_cell_2/split_1:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_2/BiasAdd_1BiasAdd$while/lstm_cell_2/MatMul_1:product:0"while/lstm_cell_2/split_1:output:1*
T0*(
_output_shapes
:???????????
while/lstm_cell_2/BiasAdd_2BiasAdd$while/lstm_cell_2/MatMul_2:product:0"while/lstm_cell_2/split_1:output:2*
T0*(
_output_shapes
:???????????
while/lstm_cell_2/BiasAdd_3BiasAdd$while/lstm_cell_2/MatMul_3:product:0"while/lstm_cell_2/split_1:output:3*
T0*(
_output_shapes
:???????????
 while/lstm_cell_2/ReadVariableOpReadVariableOp+while_lstm_cell_2_readvariableop_resource_0* 
_output_shapes
:
??*
dtype0v
%while/lstm_cell_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        x
'while/lstm_cell_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       x
'while/lstm_cell_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
while/lstm_cell_2/strided_sliceStridedSlice(while/lstm_cell_2/ReadVariableOp:value:0.while/lstm_cell_2/strided_slice/stack:output:00while/lstm_cell_2/strided_slice/stack_1:output:00while/lstm_cell_2/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
while/lstm_cell_2/MatMul_4MatMulwhile_placeholder_2(while/lstm_cell_2/strided_slice:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_2/addAddV2"while/lstm_cell_2/BiasAdd:output:0$while/lstm_cell_2/MatMul_4:product:0*
T0*(
_output_shapes
:??????????\
while/lstm_cell_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>^
while/lstm_cell_2/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ??
while/lstm_cell_2/MulMulwhile/lstm_cell_2/add:z:0 while/lstm_cell_2/Const:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_2/Add_1AddV2while/lstm_cell_2/Mul:z:0"while/lstm_cell_2/Const_1:output:0*
T0*(
_output_shapes
:??????????n
)while/lstm_cell_2/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
'while/lstm_cell_2/clip_by_value/MinimumMinimumwhile/lstm_cell_2/Add_1:z:02while/lstm_cell_2/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:??????????f
!while/lstm_cell_2/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
while/lstm_cell_2/clip_by_valueMaximum+while/lstm_cell_2/clip_by_value/Minimum:z:0*while/lstm_cell_2/clip_by_value/y:output:0*
T0*(
_output_shapes
:???????????
"while/lstm_cell_2/ReadVariableOp_1ReadVariableOp+while_lstm_cell_2_readvariableop_resource_0* 
_output_shapes
:
??*
dtype0x
'while/lstm_cell_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       z
)while/lstm_cell_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       z
)while/lstm_cell_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
!while/lstm_cell_2/strided_slice_1StridedSlice*while/lstm_cell_2/ReadVariableOp_1:value:00while/lstm_cell_2/strided_slice_1/stack:output:02while/lstm_cell_2/strided_slice_1/stack_1:output:02while/lstm_cell_2/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
while/lstm_cell_2/MatMul_5MatMulwhile_placeholder_2*while/lstm_cell_2/strided_slice_1:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_2/add_2AddV2$while/lstm_cell_2/BiasAdd_1:output:0$while/lstm_cell_2/MatMul_5:product:0*
T0*(
_output_shapes
:??????????^
while/lstm_cell_2/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>^
while/lstm_cell_2/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ??
while/lstm_cell_2/Mul_1Mulwhile/lstm_cell_2/add_2:z:0"while/lstm_cell_2/Const_2:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_2/Add_3AddV2while/lstm_cell_2/Mul_1:z:0"while/lstm_cell_2/Const_3:output:0*
T0*(
_output_shapes
:??????????p
+while/lstm_cell_2/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
)while/lstm_cell_2/clip_by_value_1/MinimumMinimumwhile/lstm_cell_2/Add_3:z:04while/lstm_cell_2/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:??????????h
#while/lstm_cell_2/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
!while/lstm_cell_2/clip_by_value_1Maximum-while/lstm_cell_2/clip_by_value_1/Minimum:z:0,while/lstm_cell_2/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_2/mul_2Mul%while/lstm_cell_2/clip_by_value_1:z:0while_placeholder_3*
T0*(
_output_shapes
:???????????
"while/lstm_cell_2/ReadVariableOp_2ReadVariableOp+while_lstm_cell_2_readvariableop_resource_0* 
_output_shapes
:
??*
dtype0x
'while/lstm_cell_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       z
)while/lstm_cell_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       z
)while/lstm_cell_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
!while/lstm_cell_2/strided_slice_2StridedSlice*while/lstm_cell_2/ReadVariableOp_2:value:00while/lstm_cell_2/strided_slice_2/stack:output:02while/lstm_cell_2/strided_slice_2/stack_1:output:02while/lstm_cell_2/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
while/lstm_cell_2/MatMul_6MatMulwhile_placeholder_2*while/lstm_cell_2/strided_slice_2:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_2/add_4AddV2$while/lstm_cell_2/BiasAdd_2:output:0$while/lstm_cell_2/MatMul_6:product:0*
T0*(
_output_shapes
:??????????n
while/lstm_cell_2/TanhTanhwhile/lstm_cell_2/add_4:z:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_2/mul_3Mul#while/lstm_cell_2/clip_by_value:z:0while/lstm_cell_2/Tanh:y:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_2/add_5AddV2while/lstm_cell_2/mul_2:z:0while/lstm_cell_2/mul_3:z:0*
T0*(
_output_shapes
:???????????
"while/lstm_cell_2/ReadVariableOp_3ReadVariableOp+while_lstm_cell_2_readvariableop_resource_0* 
_output_shapes
:
??*
dtype0x
'while/lstm_cell_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       z
)while/lstm_cell_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        z
)while/lstm_cell_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
!while/lstm_cell_2/strided_slice_3StridedSlice*while/lstm_cell_2/ReadVariableOp_3:value:00while/lstm_cell_2/strided_slice_3/stack:output:02while/lstm_cell_2/strided_slice_3/stack_1:output:02while/lstm_cell_2/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
while/lstm_cell_2/MatMul_7MatMulwhile_placeholder_2*while/lstm_cell_2/strided_slice_3:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_2/add_6AddV2$while/lstm_cell_2/BiasAdd_3:output:0$while/lstm_cell_2/MatMul_7:product:0*
T0*(
_output_shapes
:??????????^
while/lstm_cell_2/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>^
while/lstm_cell_2/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ??
while/lstm_cell_2/Mul_4Mulwhile/lstm_cell_2/add_6:z:0"while/lstm_cell_2/Const_4:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_2/Add_7AddV2while/lstm_cell_2/Mul_4:z:0"while/lstm_cell_2/Const_5:output:0*
T0*(
_output_shapes
:??????????p
+while/lstm_cell_2/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
)while/lstm_cell_2/clip_by_value_2/MinimumMinimumwhile/lstm_cell_2/Add_7:z:04while/lstm_cell_2/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:??????????h
#while/lstm_cell_2/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
!while/lstm_cell_2/clip_by_value_2Maximum-while/lstm_cell_2/clip_by_value_2/Minimum:z:0,while/lstm_cell_2/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:??????????p
while/lstm_cell_2/Tanh_1Tanhwhile/lstm_cell_2/add_5:z:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_2/mul_5Mul%while/lstm_cell_2/clip_by_value_2:z:0while/lstm_cell_2/Tanh_1:y:0*
T0*(
_output_shapes
:???????????
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_2/mul_5:z:0*
_output_shapes
: *
element_dtype0:???M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: y
while/Identity_4Identitywhile/lstm_cell_2/mul_5:z:0^while/NoOp*
T0*(
_output_shapes
:??????????y
while/Identity_5Identitywhile/lstm_cell_2/add_5:z:0^while/NoOp*
T0*(
_output_shapes
:???????????

while/NoOpNoOp!^while/lstm_cell_2/ReadVariableOp#^while/lstm_cell_2/ReadVariableOp_1#^while/lstm_cell_2/ReadVariableOp_2#^while/lstm_cell_2/ReadVariableOp_3'^while/lstm_cell_2/split/ReadVariableOp)^while/lstm_cell_2/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"X
)while_lstm_cell_2_readvariableop_resource+while_lstm_cell_2_readvariableop_resource_0"h
1while_lstm_cell_2_split_1_readvariableop_resource3while_lstm_cell_2_split_1_readvariableop_resource_0"d
/while_lstm_cell_2_split_readvariableop_resource1while_lstm_cell_2_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2D
 while/lstm_cell_2/ReadVariableOp while/lstm_cell_2/ReadVariableOp2H
"while/lstm_cell_2/ReadVariableOp_1"while/lstm_cell_2/ReadVariableOp_12H
"while/lstm_cell_2/ReadVariableOp_2"while/lstm_cell_2/ReadVariableOp_22H
"while/lstm_cell_2/ReadVariableOp_3"while/lstm_cell_2/ReadVariableOp_32P
&while/lstm_cell_2/split/ReadVariableOp&while/lstm_cell_2/split/ReadVariableOp2T
(while/lstm_cell_2/split_1/ReadVariableOp(while/lstm_cell_2/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?	
?
lstm_3_while_cond_56594*
&lstm_3_while_lstm_3_while_loop_counter0
,lstm_3_while_lstm_3_while_maximum_iterations
lstm_3_while_placeholder
lstm_3_while_placeholder_1
lstm_3_while_placeholder_2
lstm_3_while_placeholder_3,
(lstm_3_while_less_lstm_3_strided_slice_1A
=lstm_3_while_lstm_3_while_cond_56594___redundant_placeholder0A
=lstm_3_while_lstm_3_while_cond_56594___redundant_placeholder1A
=lstm_3_while_lstm_3_while_cond_56594___redundant_placeholder2A
=lstm_3_while_lstm_3_while_cond_56594___redundant_placeholder3
lstm_3_while_identity
~
lstm_3/while/LessLesslstm_3_while_placeholder(lstm_3_while_less_lstm_3_strided_slice_1*
T0*
_output_shapes
: Y
lstm_3/while/IdentityIdentitylstm_3/while/Less:z:0*
T0
*
_output_shapes
: "7
lstm_3_while_identitylstm_3/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?
?
&__inference_lstm_2_layer_call_fn_58367
inputs_0
unknown:	?
	unknown_0:	?
	unknown_1:
??
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:???????????????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_lstm_2_layer_call_and_return_conditional_losses_54397}
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:???????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :??????????????????
"
_user_specified_name
inputs/0
?
?
$sequential_1_lstm_2_while_cond_53537D
@sequential_1_lstm_2_while_sequential_1_lstm_2_while_loop_counterJ
Fsequential_1_lstm_2_while_sequential_1_lstm_2_while_maximum_iterations)
%sequential_1_lstm_2_while_placeholder+
'sequential_1_lstm_2_while_placeholder_1+
'sequential_1_lstm_2_while_placeholder_2+
'sequential_1_lstm_2_while_placeholder_3F
Bsequential_1_lstm_2_while_less_sequential_1_lstm_2_strided_slice_1[
Wsequential_1_lstm_2_while_sequential_1_lstm_2_while_cond_53537___redundant_placeholder0[
Wsequential_1_lstm_2_while_sequential_1_lstm_2_while_cond_53537___redundant_placeholder1[
Wsequential_1_lstm_2_while_sequential_1_lstm_2_while_cond_53537___redundant_placeholder2[
Wsequential_1_lstm_2_while_sequential_1_lstm_2_while_cond_53537___redundant_placeholder3&
"sequential_1_lstm_2_while_identity
?
sequential_1/lstm_2/while/LessLess%sequential_1_lstm_2_while_placeholderBsequential_1_lstm_2_while_less_sequential_1_lstm_2_strided_slice_1*
T0*
_output_shapes
: s
"sequential_1/lstm_2/while/IdentityIdentity"sequential_1/lstm_2/while/Less:z:0*
T0
*
_output_shapes
: "Q
"sequential_1_lstm_2_while_identity+sequential_1/lstm_2/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?

?
B__inference_dense_2_layer_call_and_return_conditional_losses_55412

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?7
?
A__inference_lstm_2_layer_call_and_return_conditional_losses_54397

inputs$
lstm_cell_2_54316:	? 
lstm_cell_2_54318:	?%
lstm_cell_2_54320:
??
identity??#lstm_cell_2/StatefulPartitionedCall?while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????S
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask?
#lstm_cell_2/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_2_54316lstm_cell_2_54318lstm_cell_2_54320*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:??????????:??????????:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_lstm_cell_2_layer_call_and_return_conditional_losses_54270n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_2_54316lstm_cell_2_54318lstm_cell_2_54320*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_54329*
condR
while_cond_54328*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:???????????????????*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:???????????????????l
IdentityIdentitytranspose_1:y:0^NoOp*
T0*5
_output_shapes#
!:???????????????????t
NoOpNoOp$^lstm_cell_2/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????: : : 2J
#lstm_cell_2/StatefulPartitionedCall#lstm_cell_2/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
??
?
A__inference_lstm_3_layer_call_and_return_conditional_losses_55393

inputs=
)lstm_cell_3_split_readvariableop_resource:
??:
+lstm_cell_3_split_1_readvariableop_resource:	?7
#lstm_cell_3_readvariableop_resource:
??
identity??lstm_cell_3/ReadVariableOp?lstm_cell_3/ReadVariableOp_1?lstm_cell_3/ReadVariableOp_2?lstm_cell_3/ReadVariableOp_3? lstm_cell_3/split/ReadVariableOp?"lstm_cell_3/split_1/ReadVariableOp?while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????S
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:??????????D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask]
lstm_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
 lstm_cell_3/split/ReadVariableOpReadVariableOp)lstm_cell_3_split_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
lstm_cell_3/splitSplit$lstm_cell_3/split/split_dim:output:0(lstm_cell_3/split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
??:
??:
??:
??*
	num_split?
lstm_cell_3/MatMulMatMulstrided_slice_2:output:0lstm_cell_3/split:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_3/MatMul_1MatMulstrided_slice_2:output:0lstm_cell_3/split:output:1*
T0*(
_output_shapes
:???????????
lstm_cell_3/MatMul_2MatMulstrided_slice_2:output:0lstm_cell_3/split:output:2*
T0*(
_output_shapes
:???????????
lstm_cell_3/MatMul_3MatMulstrided_slice_2:output:0lstm_cell_3/split:output:3*
T0*(
_output_shapes
:??????????_
lstm_cell_3/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
"lstm_cell_3/split_1/ReadVariableOpReadVariableOp+lstm_cell_3_split_1_readvariableop_resource*
_output_shapes	
:?*
dtype0?
lstm_cell_3/split_1Split&lstm_cell_3/split_1/split_dim:output:0*lstm_cell_3/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split?
lstm_cell_3/BiasAddBiasAddlstm_cell_3/MatMul:product:0lstm_cell_3/split_1:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_3/BiasAdd_1BiasAddlstm_cell_3/MatMul_1:product:0lstm_cell_3/split_1:output:1*
T0*(
_output_shapes
:???????????
lstm_cell_3/BiasAdd_2BiasAddlstm_cell_3/MatMul_2:product:0lstm_cell_3/split_1:output:2*
T0*(
_output_shapes
:???????????
lstm_cell_3/BiasAdd_3BiasAddlstm_cell_3/MatMul_3:product:0lstm_cell_3/split_1:output:3*
T0*(
_output_shapes
:???????????
lstm_cell_3/ReadVariableOpReadVariableOp#lstm_cell_3_readvariableop_resource* 
_output_shapes
:
??*
dtype0p
lstm_cell_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        r
!lstm_cell_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   r
!lstm_cell_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
lstm_cell_3/strided_sliceStridedSlice"lstm_cell_3/ReadVariableOp:value:0(lstm_cell_3/strided_slice/stack:output:0*lstm_cell_3/strided_slice/stack_1:output:0*lstm_cell_3/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
lstm_cell_3/MatMul_4MatMulzeros:output:0"lstm_cell_3/strided_slice:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_3/addAddV2lstm_cell_3/BiasAdd:output:0lstm_cell_3/MatMul_4:product:0*
T0*(
_output_shapes
:??????????V
lstm_cell_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>X
lstm_cell_3/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?z
lstm_cell_3/MulMullstm_cell_3/add:z:0lstm_cell_3/Const:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_3/Add_1AddV2lstm_cell_3/Mul:z:0lstm_cell_3/Const_1:output:0*
T0*(
_output_shapes
:??????????h
#lstm_cell_3/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
!lstm_cell_3/clip_by_value/MinimumMinimumlstm_cell_3/Add_1:z:0,lstm_cell_3/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:??????????`
lstm_cell_3/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
lstm_cell_3/clip_by_valueMaximum%lstm_cell_3/clip_by_value/Minimum:z:0$lstm_cell_3/clip_by_value/y:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_3/ReadVariableOp_1ReadVariableOp#lstm_cell_3_readvariableop_resource* 
_output_shapes
:
??*
dtype0r
!lstm_cell_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   t
#lstm_cell_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       t
#lstm_cell_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
lstm_cell_3/strided_slice_1StridedSlice$lstm_cell_3/ReadVariableOp_1:value:0*lstm_cell_3/strided_slice_1/stack:output:0,lstm_cell_3/strided_slice_1/stack_1:output:0,lstm_cell_3/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
lstm_cell_3/MatMul_5MatMulzeros:output:0$lstm_cell_3/strided_slice_1:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_3/add_2AddV2lstm_cell_3/BiasAdd_1:output:0lstm_cell_3/MatMul_5:product:0*
T0*(
_output_shapes
:??????????X
lstm_cell_3/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>X
lstm_cell_3/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ??
lstm_cell_3/Mul_1Mullstm_cell_3/add_2:z:0lstm_cell_3/Const_2:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_3/Add_3AddV2lstm_cell_3/Mul_1:z:0lstm_cell_3/Const_3:output:0*
T0*(
_output_shapes
:??????????j
%lstm_cell_3/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
#lstm_cell_3/clip_by_value_1/MinimumMinimumlstm_cell_3/Add_3:z:0.lstm_cell_3/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:??????????b
lstm_cell_3/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
lstm_cell_3/clip_by_value_1Maximum'lstm_cell_3/clip_by_value_1/Minimum:z:0&lstm_cell_3/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:??????????~
lstm_cell_3/mul_2Mullstm_cell_3/clip_by_value_1:z:0zeros_1:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_3/ReadVariableOp_2ReadVariableOp#lstm_cell_3_readvariableop_resource* 
_output_shapes
:
??*
dtype0r
!lstm_cell_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       t
#lstm_cell_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  t
#lstm_cell_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
lstm_cell_3/strided_slice_2StridedSlice$lstm_cell_3/ReadVariableOp_2:value:0*lstm_cell_3/strided_slice_2/stack:output:0,lstm_cell_3/strided_slice_2/stack_1:output:0,lstm_cell_3/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
lstm_cell_3/MatMul_6MatMulzeros:output:0$lstm_cell_3/strided_slice_2:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_3/add_4AddV2lstm_cell_3/BiasAdd_2:output:0lstm_cell_3/MatMul_6:product:0*
T0*(
_output_shapes
:??????????b
lstm_cell_3/TanhTanhlstm_cell_3/add_4:z:0*
T0*(
_output_shapes
:???????????
lstm_cell_3/mul_3Mullstm_cell_3/clip_by_value:z:0lstm_cell_3/Tanh:y:0*
T0*(
_output_shapes
:??????????{
lstm_cell_3/add_5AddV2lstm_cell_3/mul_2:z:0lstm_cell_3/mul_3:z:0*
T0*(
_output_shapes
:???????????
lstm_cell_3/ReadVariableOp_3ReadVariableOp#lstm_cell_3_readvariableop_resource* 
_output_shapes
:
??*
dtype0r
!lstm_cell_3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  t
#lstm_cell_3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        t
#lstm_cell_3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
lstm_cell_3/strided_slice_3StridedSlice$lstm_cell_3/ReadVariableOp_3:value:0*lstm_cell_3/strided_slice_3/stack:output:0,lstm_cell_3/strided_slice_3/stack_1:output:0,lstm_cell_3/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
lstm_cell_3/MatMul_7MatMulzeros:output:0$lstm_cell_3/strided_slice_3:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_3/add_6AddV2lstm_cell_3/BiasAdd_3:output:0lstm_cell_3/MatMul_7:product:0*
T0*(
_output_shapes
:??????????X
lstm_cell_3/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>X
lstm_cell_3/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ??
lstm_cell_3/Mul_4Mullstm_cell_3/add_6:z:0lstm_cell_3/Const_4:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_3/Add_7AddV2lstm_cell_3/Mul_4:z:0lstm_cell_3/Const_5:output:0*
T0*(
_output_shapes
:??????????j
%lstm_cell_3/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
#lstm_cell_3/clip_by_value_2/MinimumMinimumlstm_cell_3/Add_7:z:0.lstm_cell_3/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:??????????b
lstm_cell_3/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
lstm_cell_3/clip_by_value_2Maximum'lstm_cell_3/clip_by_value_2/Minimum:z:0&lstm_cell_3/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:??????????d
lstm_cell_3/Tanh_1Tanhlstm_cell_3/add_5:z:0*
T0*(
_output_shapes
:???????????
lstm_cell_3/mul_5Mullstm_cell_3/clip_by_value_2:z:0lstm_cell_3/Tanh_1:y:0*
T0*(
_output_shapes
:??????????n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   ?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)lstm_cell_3_split_readvariableop_resource+lstm_cell_3_split_1_readvariableop_resource#lstm_cell_3_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_55253*
condR
while_cond_55252*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????h
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp^lstm_cell_3/ReadVariableOp^lstm_cell_3/ReadVariableOp_1^lstm_cell_3/ReadVariableOp_2^lstm_cell_3/ReadVariableOp_3!^lstm_cell_3/split/ReadVariableOp#^lstm_cell_3/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:??????????: : : 28
lstm_cell_3/ReadVariableOplstm_cell_3/ReadVariableOp2<
lstm_cell_3/ReadVariableOp_1lstm_cell_3/ReadVariableOp_12<
lstm_cell_3/ReadVariableOp_2lstm_cell_3/ReadVariableOp_22<
lstm_cell_3/ReadVariableOp_3lstm_cell_3/ReadVariableOp_32D
 lstm_cell_3/split/ReadVariableOp lstm_cell_3/split/ReadVariableOp2H
"lstm_cell_3/split_1/ReadVariableOp"lstm_cell_3/split_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
while_cond_58204
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_58204___redundant_placeholder03
/while_while_cond_58204___redundant_placeholder13
/while_while_cond_58204___redundant_placeholder23
/while_while_cond_58204___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?
?
+__inference_lstm_cell_3_layer_call_fn_59921

inputs
states_0
states_1
unknown:
??
	unknown_0:	?
	unknown_1:
??
identity

identity_1

identity_2??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:??????????:??????????:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_lstm_cell_3_layer_call_and_return_conditional_losses_54732p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????r

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:??????????r

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:??????????:??????????:??????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/0:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/1
?
?
G__inference_sequential_1_layer_call_and_return_conditional_losses_56090

inputs
lstm_2_56065:	?
lstm_2_56067:	? 
lstm_2_56069:
?? 
lstm_3_56072:
??
lstm_3_56074:	? 
lstm_3_56076:
??!
dense_2_56079:
??
dense_2_56081:	? 
dense_3_56084:	?
dense_3_56086:
identity??dense_2/StatefulPartitionedCall?dense_3/StatefulPartitionedCall?lstm_2/StatefulPartitionedCall?lstm_3/StatefulPartitionedCall?
lstm_2/StatefulPartitionedCallStatefulPartitionedCallinputslstm_2_56065lstm_2_56067lstm_2_56069*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_lstm_2_layer_call_and_return_conditional_losses_56026?
lstm_3/StatefulPartitionedCallStatefulPartitionedCall'lstm_2/StatefulPartitionedCall:output:0lstm_3_56072lstm_3_56074lstm_3_56076*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_lstm_3_layer_call_and_return_conditional_losses_55748?
dense_2/StatefulPartitionedCallStatefulPartitionedCall'lstm_3/StatefulPartitionedCall:output:0dense_2_56079dense_2_56081*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_55412?
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_56084dense_3_56086*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_55429w
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall^lstm_2/StatefulPartitionedCall^lstm_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????: : : : : : : : : : 2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2@
lstm_2/StatefulPartitionedCalllstm_2/StatefulPartitionedCall2@
lstm_3/StatefulPartitionedCalllstm_3/StatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?{
?	
while_body_54990
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_lstm_cell_2_split_readvariableop_resource_0:	?B
3while_lstm_cell_2_split_1_readvariableop_resource_0:	??
+while_lstm_cell_2_readvariableop_resource_0:
??
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_lstm_cell_2_split_readvariableop_resource:	?@
1while_lstm_cell_2_split_1_readvariableop_resource:	?=
)while_lstm_cell_2_readvariableop_resource:
???? while/lstm_cell_2/ReadVariableOp?"while/lstm_cell_2/ReadVariableOp_1?"while/lstm_cell_2/ReadVariableOp_2?"while/lstm_cell_2/ReadVariableOp_3?&while/lstm_cell_2/split/ReadVariableOp?(while/lstm_cell_2/split_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0c
!while/lstm_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
&while/lstm_cell_2/split/ReadVariableOpReadVariableOp1while_lstm_cell_2_split_readvariableop_resource_0*
_output_shapes
:	?*
dtype0?
while/lstm_cell_2/splitSplit*while/lstm_cell_2/split/split_dim:output:0.while/lstm_cell_2/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	?:	?:	?:	?*
	num_split?
while/lstm_cell_2/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_2/split:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_2/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_2/split:output:1*
T0*(
_output_shapes
:???????????
while/lstm_cell_2/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_2/split:output:2*
T0*(
_output_shapes
:???????????
while/lstm_cell_2/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_2/split:output:3*
T0*(
_output_shapes
:??????????e
#while/lstm_cell_2/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
(while/lstm_cell_2/split_1/ReadVariableOpReadVariableOp3while_lstm_cell_2_split_1_readvariableop_resource_0*
_output_shapes	
:?*
dtype0?
while/lstm_cell_2/split_1Split,while/lstm_cell_2/split_1/split_dim:output:00while/lstm_cell_2/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split?
while/lstm_cell_2/BiasAddBiasAdd"while/lstm_cell_2/MatMul:product:0"while/lstm_cell_2/split_1:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_2/BiasAdd_1BiasAdd$while/lstm_cell_2/MatMul_1:product:0"while/lstm_cell_2/split_1:output:1*
T0*(
_output_shapes
:???????????
while/lstm_cell_2/BiasAdd_2BiasAdd$while/lstm_cell_2/MatMul_2:product:0"while/lstm_cell_2/split_1:output:2*
T0*(
_output_shapes
:???????????
while/lstm_cell_2/BiasAdd_3BiasAdd$while/lstm_cell_2/MatMul_3:product:0"while/lstm_cell_2/split_1:output:3*
T0*(
_output_shapes
:???????????
 while/lstm_cell_2/ReadVariableOpReadVariableOp+while_lstm_cell_2_readvariableop_resource_0* 
_output_shapes
:
??*
dtype0v
%while/lstm_cell_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        x
'while/lstm_cell_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       x
'while/lstm_cell_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
while/lstm_cell_2/strided_sliceStridedSlice(while/lstm_cell_2/ReadVariableOp:value:0.while/lstm_cell_2/strided_slice/stack:output:00while/lstm_cell_2/strided_slice/stack_1:output:00while/lstm_cell_2/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
while/lstm_cell_2/MatMul_4MatMulwhile_placeholder_2(while/lstm_cell_2/strided_slice:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_2/addAddV2"while/lstm_cell_2/BiasAdd:output:0$while/lstm_cell_2/MatMul_4:product:0*
T0*(
_output_shapes
:??????????\
while/lstm_cell_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>^
while/lstm_cell_2/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ??
while/lstm_cell_2/MulMulwhile/lstm_cell_2/add:z:0 while/lstm_cell_2/Const:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_2/Add_1AddV2while/lstm_cell_2/Mul:z:0"while/lstm_cell_2/Const_1:output:0*
T0*(
_output_shapes
:??????????n
)while/lstm_cell_2/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
'while/lstm_cell_2/clip_by_value/MinimumMinimumwhile/lstm_cell_2/Add_1:z:02while/lstm_cell_2/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:??????????f
!while/lstm_cell_2/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
while/lstm_cell_2/clip_by_valueMaximum+while/lstm_cell_2/clip_by_value/Minimum:z:0*while/lstm_cell_2/clip_by_value/y:output:0*
T0*(
_output_shapes
:???????????
"while/lstm_cell_2/ReadVariableOp_1ReadVariableOp+while_lstm_cell_2_readvariableop_resource_0* 
_output_shapes
:
??*
dtype0x
'while/lstm_cell_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       z
)while/lstm_cell_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       z
)while/lstm_cell_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
!while/lstm_cell_2/strided_slice_1StridedSlice*while/lstm_cell_2/ReadVariableOp_1:value:00while/lstm_cell_2/strided_slice_1/stack:output:02while/lstm_cell_2/strided_slice_1/stack_1:output:02while/lstm_cell_2/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
while/lstm_cell_2/MatMul_5MatMulwhile_placeholder_2*while/lstm_cell_2/strided_slice_1:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_2/add_2AddV2$while/lstm_cell_2/BiasAdd_1:output:0$while/lstm_cell_2/MatMul_5:product:0*
T0*(
_output_shapes
:??????????^
while/lstm_cell_2/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>^
while/lstm_cell_2/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ??
while/lstm_cell_2/Mul_1Mulwhile/lstm_cell_2/add_2:z:0"while/lstm_cell_2/Const_2:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_2/Add_3AddV2while/lstm_cell_2/Mul_1:z:0"while/lstm_cell_2/Const_3:output:0*
T0*(
_output_shapes
:??????????p
+while/lstm_cell_2/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
)while/lstm_cell_2/clip_by_value_1/MinimumMinimumwhile/lstm_cell_2/Add_3:z:04while/lstm_cell_2/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:??????????h
#while/lstm_cell_2/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
!while/lstm_cell_2/clip_by_value_1Maximum-while/lstm_cell_2/clip_by_value_1/Minimum:z:0,while/lstm_cell_2/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_2/mul_2Mul%while/lstm_cell_2/clip_by_value_1:z:0while_placeholder_3*
T0*(
_output_shapes
:???????????
"while/lstm_cell_2/ReadVariableOp_2ReadVariableOp+while_lstm_cell_2_readvariableop_resource_0* 
_output_shapes
:
??*
dtype0x
'while/lstm_cell_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       z
)while/lstm_cell_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       z
)while/lstm_cell_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
!while/lstm_cell_2/strided_slice_2StridedSlice*while/lstm_cell_2/ReadVariableOp_2:value:00while/lstm_cell_2/strided_slice_2/stack:output:02while/lstm_cell_2/strided_slice_2/stack_1:output:02while/lstm_cell_2/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
while/lstm_cell_2/MatMul_6MatMulwhile_placeholder_2*while/lstm_cell_2/strided_slice_2:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_2/add_4AddV2$while/lstm_cell_2/BiasAdd_2:output:0$while/lstm_cell_2/MatMul_6:product:0*
T0*(
_output_shapes
:??????????n
while/lstm_cell_2/TanhTanhwhile/lstm_cell_2/add_4:z:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_2/mul_3Mul#while/lstm_cell_2/clip_by_value:z:0while/lstm_cell_2/Tanh:y:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_2/add_5AddV2while/lstm_cell_2/mul_2:z:0while/lstm_cell_2/mul_3:z:0*
T0*(
_output_shapes
:???????????
"while/lstm_cell_2/ReadVariableOp_3ReadVariableOp+while_lstm_cell_2_readvariableop_resource_0* 
_output_shapes
:
??*
dtype0x
'while/lstm_cell_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       z
)while/lstm_cell_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        z
)while/lstm_cell_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
!while/lstm_cell_2/strided_slice_3StridedSlice*while/lstm_cell_2/ReadVariableOp_3:value:00while/lstm_cell_2/strided_slice_3/stack:output:02while/lstm_cell_2/strided_slice_3/stack_1:output:02while/lstm_cell_2/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
while/lstm_cell_2/MatMul_7MatMulwhile_placeholder_2*while/lstm_cell_2/strided_slice_3:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_2/add_6AddV2$while/lstm_cell_2/BiasAdd_3:output:0$while/lstm_cell_2/MatMul_7:product:0*
T0*(
_output_shapes
:??????????^
while/lstm_cell_2/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>^
while/lstm_cell_2/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ??
while/lstm_cell_2/Mul_4Mulwhile/lstm_cell_2/add_6:z:0"while/lstm_cell_2/Const_4:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_2/Add_7AddV2while/lstm_cell_2/Mul_4:z:0"while/lstm_cell_2/Const_5:output:0*
T0*(
_output_shapes
:??????????p
+while/lstm_cell_2/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
)while/lstm_cell_2/clip_by_value_2/MinimumMinimumwhile/lstm_cell_2/Add_7:z:04while/lstm_cell_2/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:??????????h
#while/lstm_cell_2/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
!while/lstm_cell_2/clip_by_value_2Maximum-while/lstm_cell_2/clip_by_value_2/Minimum:z:0,while/lstm_cell_2/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:??????????p
while/lstm_cell_2/Tanh_1Tanhwhile/lstm_cell_2/add_5:z:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_2/mul_5Mul%while/lstm_cell_2/clip_by_value_2:z:0while/lstm_cell_2/Tanh_1:y:0*
T0*(
_output_shapes
:???????????
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_2/mul_5:z:0*
_output_shapes
: *
element_dtype0:???M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: y
while/Identity_4Identitywhile/lstm_cell_2/mul_5:z:0^while/NoOp*
T0*(
_output_shapes
:??????????y
while/Identity_5Identitywhile/lstm_cell_2/add_5:z:0^while/NoOp*
T0*(
_output_shapes
:???????????

while/NoOpNoOp!^while/lstm_cell_2/ReadVariableOp#^while/lstm_cell_2/ReadVariableOp_1#^while/lstm_cell_2/ReadVariableOp_2#^while/lstm_cell_2/ReadVariableOp_3'^while/lstm_cell_2/split/ReadVariableOp)^while/lstm_cell_2/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"X
)while_lstm_cell_2_readvariableop_resource+while_lstm_cell_2_readvariableop_resource_0"h
1while_lstm_cell_2_split_1_readvariableop_resource3while_lstm_cell_2_split_1_readvariableop_resource_0"d
/while_lstm_cell_2_split_readvariableop_resource1while_lstm_cell_2_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2D
 while/lstm_cell_2/ReadVariableOp while/lstm_cell_2/ReadVariableOp2H
"while/lstm_cell_2/ReadVariableOp_1"while/lstm_cell_2/ReadVariableOp_12H
"while/lstm_cell_2/ReadVariableOp_2"while/lstm_cell_2/ReadVariableOp_22H
"while/lstm_cell_2/ReadVariableOp_3"while/lstm_cell_2/ReadVariableOp_32P
&while/lstm_cell_2/split/ReadVariableOp&while/lstm_cell_2/split/ReadVariableOp2T
(while/lstm_cell_2/split_1/ReadVariableOp(while/lstm_cell_2/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
??
?

G__inference_sequential_1_layer_call_and_return_conditional_losses_57271

inputsC
0lstm_2_lstm_cell_2_split_readvariableop_resource:	?A
2lstm_2_lstm_cell_2_split_1_readvariableop_resource:	?>
*lstm_2_lstm_cell_2_readvariableop_resource:
??D
0lstm_3_lstm_cell_3_split_readvariableop_resource:
??A
2lstm_3_lstm_cell_3_split_1_readvariableop_resource:	?>
*lstm_3_lstm_cell_3_readvariableop_resource:
??:
&dense_2_matmul_readvariableop_resource:
??6
'dense_2_biasadd_readvariableop_resource:	?9
&dense_3_matmul_readvariableop_resource:	?5
'dense_3_biasadd_readvariableop_resource:
identity??dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOp?dense_3/BiasAdd/ReadVariableOp?dense_3/MatMul/ReadVariableOp?!lstm_2/lstm_cell_2/ReadVariableOp?#lstm_2/lstm_cell_2/ReadVariableOp_1?#lstm_2/lstm_cell_2/ReadVariableOp_2?#lstm_2/lstm_cell_2/ReadVariableOp_3?'lstm_2/lstm_cell_2/split/ReadVariableOp?)lstm_2/lstm_cell_2/split_1/ReadVariableOp?lstm_2/while?!lstm_3/lstm_cell_3/ReadVariableOp?#lstm_3/lstm_cell_3/ReadVariableOp_1?#lstm_3/lstm_cell_3/ReadVariableOp_2?#lstm_3/lstm_cell_3/ReadVariableOp_3?'lstm_3/lstm_cell_3/split/ReadVariableOp?)lstm_3/lstm_cell_3/split_1/ReadVariableOp?lstm_3/whileB
lstm_2/ShapeShapeinputs*
T0*
_output_shapes
:d
lstm_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: f
lstm_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:f
lstm_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
lstm_2/strided_sliceStridedSlicelstm_2/Shape:output:0#lstm_2/strided_slice/stack:output:0%lstm_2/strided_slice/stack_1:output:0%lstm_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
lstm_2/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :??
lstm_2/zeros/packedPacklstm_2/strided_slice:output:0lstm_2/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:W
lstm_2/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
lstm_2/zerosFilllstm_2/zeros/packed:output:0lstm_2/zeros/Const:output:0*
T0*(
_output_shapes
:??????????Z
lstm_2/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :??
lstm_2/zeros_1/packedPacklstm_2/strided_slice:output:0 lstm_2/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:Y
lstm_2/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
lstm_2/zeros_1Filllstm_2/zeros_1/packed:output:0lstm_2/zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????j
lstm_2/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          {
lstm_2/transpose	Transposeinputslstm_2/transpose/perm:output:0*
T0*+
_output_shapes
:?????????R
lstm_2/Shape_1Shapelstm_2/transpose:y:0*
T0*
_output_shapes
:f
lstm_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: h
lstm_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
lstm_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
lstm_2/strided_slice_1StridedSlicelstm_2/Shape_1:output:0%lstm_2/strided_slice_1/stack:output:0'lstm_2/strided_slice_1/stack_1:output:0'lstm_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskm
"lstm_2/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
lstm_2/TensorArrayV2TensorListReserve+lstm_2/TensorArrayV2/element_shape:output:0lstm_2/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
<lstm_2/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
.lstm_2/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_2/transpose:y:0Elstm_2/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???f
lstm_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: h
lstm_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
lstm_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
lstm_2/strided_slice_2StridedSlicelstm_2/transpose:y:0%lstm_2/strided_slice_2/stack:output:0'lstm_2/strided_slice_2/stack_1:output:0'lstm_2/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_maskd
"lstm_2/lstm_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
'lstm_2/lstm_cell_2/split/ReadVariableOpReadVariableOp0lstm_2_lstm_cell_2_split_readvariableop_resource*
_output_shapes
:	?*
dtype0?
lstm_2/lstm_cell_2/splitSplit+lstm_2/lstm_cell_2/split/split_dim:output:0/lstm_2/lstm_cell_2/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	?:	?:	?:	?*
	num_split?
lstm_2/lstm_cell_2/MatMulMatMullstm_2/strided_slice_2:output:0!lstm_2/lstm_cell_2/split:output:0*
T0*(
_output_shapes
:???????????
lstm_2/lstm_cell_2/MatMul_1MatMullstm_2/strided_slice_2:output:0!lstm_2/lstm_cell_2/split:output:1*
T0*(
_output_shapes
:???????????
lstm_2/lstm_cell_2/MatMul_2MatMullstm_2/strided_slice_2:output:0!lstm_2/lstm_cell_2/split:output:2*
T0*(
_output_shapes
:???????????
lstm_2/lstm_cell_2/MatMul_3MatMullstm_2/strided_slice_2:output:0!lstm_2/lstm_cell_2/split:output:3*
T0*(
_output_shapes
:??????????f
$lstm_2/lstm_cell_2/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
)lstm_2/lstm_cell_2/split_1/ReadVariableOpReadVariableOp2lstm_2_lstm_cell_2_split_1_readvariableop_resource*
_output_shapes	
:?*
dtype0?
lstm_2/lstm_cell_2/split_1Split-lstm_2/lstm_cell_2/split_1/split_dim:output:01lstm_2/lstm_cell_2/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split?
lstm_2/lstm_cell_2/BiasAddBiasAdd#lstm_2/lstm_cell_2/MatMul:product:0#lstm_2/lstm_cell_2/split_1:output:0*
T0*(
_output_shapes
:???????????
lstm_2/lstm_cell_2/BiasAdd_1BiasAdd%lstm_2/lstm_cell_2/MatMul_1:product:0#lstm_2/lstm_cell_2/split_1:output:1*
T0*(
_output_shapes
:???????????
lstm_2/lstm_cell_2/BiasAdd_2BiasAdd%lstm_2/lstm_cell_2/MatMul_2:product:0#lstm_2/lstm_cell_2/split_1:output:2*
T0*(
_output_shapes
:???????????
lstm_2/lstm_cell_2/BiasAdd_3BiasAdd%lstm_2/lstm_cell_2/MatMul_3:product:0#lstm_2/lstm_cell_2/split_1:output:3*
T0*(
_output_shapes
:???????????
!lstm_2/lstm_cell_2/ReadVariableOpReadVariableOp*lstm_2_lstm_cell_2_readvariableop_resource* 
_output_shapes
:
??*
dtype0w
&lstm_2/lstm_cell_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        y
(lstm_2/lstm_cell_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       y
(lstm_2/lstm_cell_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
 lstm_2/lstm_cell_2/strided_sliceStridedSlice)lstm_2/lstm_cell_2/ReadVariableOp:value:0/lstm_2/lstm_cell_2/strided_slice/stack:output:01lstm_2/lstm_cell_2/strided_slice/stack_1:output:01lstm_2/lstm_cell_2/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
lstm_2/lstm_cell_2/MatMul_4MatMullstm_2/zeros:output:0)lstm_2/lstm_cell_2/strided_slice:output:0*
T0*(
_output_shapes
:???????????
lstm_2/lstm_cell_2/addAddV2#lstm_2/lstm_cell_2/BiasAdd:output:0%lstm_2/lstm_cell_2/MatMul_4:product:0*
T0*(
_output_shapes
:??????????]
lstm_2/lstm_cell_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>_
lstm_2/lstm_cell_2/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ??
lstm_2/lstm_cell_2/MulMullstm_2/lstm_cell_2/add:z:0!lstm_2/lstm_cell_2/Const:output:0*
T0*(
_output_shapes
:???????????
lstm_2/lstm_cell_2/Add_1AddV2lstm_2/lstm_cell_2/Mul:z:0#lstm_2/lstm_cell_2/Const_1:output:0*
T0*(
_output_shapes
:??????????o
*lstm_2/lstm_cell_2/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
(lstm_2/lstm_cell_2/clip_by_value/MinimumMinimumlstm_2/lstm_cell_2/Add_1:z:03lstm_2/lstm_cell_2/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:??????????g
"lstm_2/lstm_cell_2/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
 lstm_2/lstm_cell_2/clip_by_valueMaximum,lstm_2/lstm_cell_2/clip_by_value/Minimum:z:0+lstm_2/lstm_cell_2/clip_by_value/y:output:0*
T0*(
_output_shapes
:???????????
#lstm_2/lstm_cell_2/ReadVariableOp_1ReadVariableOp*lstm_2_lstm_cell_2_readvariableop_resource* 
_output_shapes
:
??*
dtype0y
(lstm_2/lstm_cell_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       {
*lstm_2/lstm_cell_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       {
*lstm_2/lstm_cell_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
"lstm_2/lstm_cell_2/strided_slice_1StridedSlice+lstm_2/lstm_cell_2/ReadVariableOp_1:value:01lstm_2/lstm_cell_2/strided_slice_1/stack:output:03lstm_2/lstm_cell_2/strided_slice_1/stack_1:output:03lstm_2/lstm_cell_2/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
lstm_2/lstm_cell_2/MatMul_5MatMullstm_2/zeros:output:0+lstm_2/lstm_cell_2/strided_slice_1:output:0*
T0*(
_output_shapes
:???????????
lstm_2/lstm_cell_2/add_2AddV2%lstm_2/lstm_cell_2/BiasAdd_1:output:0%lstm_2/lstm_cell_2/MatMul_5:product:0*
T0*(
_output_shapes
:??????????_
lstm_2/lstm_cell_2/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>_
lstm_2/lstm_cell_2/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ??
lstm_2/lstm_cell_2/Mul_1Mullstm_2/lstm_cell_2/add_2:z:0#lstm_2/lstm_cell_2/Const_2:output:0*
T0*(
_output_shapes
:???????????
lstm_2/lstm_cell_2/Add_3AddV2lstm_2/lstm_cell_2/Mul_1:z:0#lstm_2/lstm_cell_2/Const_3:output:0*
T0*(
_output_shapes
:??????????q
,lstm_2/lstm_cell_2/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
*lstm_2/lstm_cell_2/clip_by_value_1/MinimumMinimumlstm_2/lstm_cell_2/Add_3:z:05lstm_2/lstm_cell_2/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:??????????i
$lstm_2/lstm_cell_2/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
"lstm_2/lstm_cell_2/clip_by_value_1Maximum.lstm_2/lstm_cell_2/clip_by_value_1/Minimum:z:0-lstm_2/lstm_cell_2/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:???????????
lstm_2/lstm_cell_2/mul_2Mul&lstm_2/lstm_cell_2/clip_by_value_1:z:0lstm_2/zeros_1:output:0*
T0*(
_output_shapes
:???????????
#lstm_2/lstm_cell_2/ReadVariableOp_2ReadVariableOp*lstm_2_lstm_cell_2_readvariableop_resource* 
_output_shapes
:
??*
dtype0y
(lstm_2/lstm_cell_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       {
*lstm_2/lstm_cell_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       {
*lstm_2/lstm_cell_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
"lstm_2/lstm_cell_2/strided_slice_2StridedSlice+lstm_2/lstm_cell_2/ReadVariableOp_2:value:01lstm_2/lstm_cell_2/strided_slice_2/stack:output:03lstm_2/lstm_cell_2/strided_slice_2/stack_1:output:03lstm_2/lstm_cell_2/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
lstm_2/lstm_cell_2/MatMul_6MatMullstm_2/zeros:output:0+lstm_2/lstm_cell_2/strided_slice_2:output:0*
T0*(
_output_shapes
:???????????
lstm_2/lstm_cell_2/add_4AddV2%lstm_2/lstm_cell_2/BiasAdd_2:output:0%lstm_2/lstm_cell_2/MatMul_6:product:0*
T0*(
_output_shapes
:??????????p
lstm_2/lstm_cell_2/TanhTanhlstm_2/lstm_cell_2/add_4:z:0*
T0*(
_output_shapes
:???????????
lstm_2/lstm_cell_2/mul_3Mul$lstm_2/lstm_cell_2/clip_by_value:z:0lstm_2/lstm_cell_2/Tanh:y:0*
T0*(
_output_shapes
:???????????
lstm_2/lstm_cell_2/add_5AddV2lstm_2/lstm_cell_2/mul_2:z:0lstm_2/lstm_cell_2/mul_3:z:0*
T0*(
_output_shapes
:???????????
#lstm_2/lstm_cell_2/ReadVariableOp_3ReadVariableOp*lstm_2_lstm_cell_2_readvariableop_resource* 
_output_shapes
:
??*
dtype0y
(lstm_2/lstm_cell_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       {
*lstm_2/lstm_cell_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        {
*lstm_2/lstm_cell_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
"lstm_2/lstm_cell_2/strided_slice_3StridedSlice+lstm_2/lstm_cell_2/ReadVariableOp_3:value:01lstm_2/lstm_cell_2/strided_slice_3/stack:output:03lstm_2/lstm_cell_2/strided_slice_3/stack_1:output:03lstm_2/lstm_cell_2/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
lstm_2/lstm_cell_2/MatMul_7MatMullstm_2/zeros:output:0+lstm_2/lstm_cell_2/strided_slice_3:output:0*
T0*(
_output_shapes
:???????????
lstm_2/lstm_cell_2/add_6AddV2%lstm_2/lstm_cell_2/BiasAdd_3:output:0%lstm_2/lstm_cell_2/MatMul_7:product:0*
T0*(
_output_shapes
:??????????_
lstm_2/lstm_cell_2/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>_
lstm_2/lstm_cell_2/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ??
lstm_2/lstm_cell_2/Mul_4Mullstm_2/lstm_cell_2/add_6:z:0#lstm_2/lstm_cell_2/Const_4:output:0*
T0*(
_output_shapes
:???????????
lstm_2/lstm_cell_2/Add_7AddV2lstm_2/lstm_cell_2/Mul_4:z:0#lstm_2/lstm_cell_2/Const_5:output:0*
T0*(
_output_shapes
:??????????q
,lstm_2/lstm_cell_2/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
*lstm_2/lstm_cell_2/clip_by_value_2/MinimumMinimumlstm_2/lstm_cell_2/Add_7:z:05lstm_2/lstm_cell_2/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:??????????i
$lstm_2/lstm_cell_2/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
"lstm_2/lstm_cell_2/clip_by_value_2Maximum.lstm_2/lstm_cell_2/clip_by_value_2/Minimum:z:0-lstm_2/lstm_cell_2/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:??????????r
lstm_2/lstm_cell_2/Tanh_1Tanhlstm_2/lstm_cell_2/add_5:z:0*
T0*(
_output_shapes
:???????????
lstm_2/lstm_cell_2/mul_5Mul&lstm_2/lstm_cell_2/clip_by_value_2:z:0lstm_2/lstm_cell_2/Tanh_1:y:0*
T0*(
_output_shapes
:??????????u
$lstm_2/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
lstm_2/TensorArrayV2_1TensorListReserve-lstm_2/TensorArrayV2_1/element_shape:output:0lstm_2/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???M
lstm_2/timeConst*
_output_shapes
: *
dtype0*
value	B : j
lstm_2/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????[
lstm_2/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
lstm_2/whileWhile"lstm_2/while/loop_counter:output:0(lstm_2/while/maximum_iterations:output:0lstm_2/time:output:0lstm_2/TensorArrayV2_1:handle:0lstm_2/zeros:output:0lstm_2/zeros_1:output:0lstm_2/strided_slice_1:output:0>lstm_2/TensorArrayUnstack/TensorListFromTensor:output_handle:00lstm_2_lstm_cell_2_split_readvariableop_resource2lstm_2_lstm_cell_2_split_1_readvariableop_resource*lstm_2_lstm_cell_2_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *#
bodyR
lstm_2_while_body_56865*#
condR
lstm_2_while_cond_56864*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations ?
7lstm_2/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
)lstm_2/TensorArrayV2Stack/TensorListStackTensorListStacklstm_2/while:output:3@lstm_2/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
element_dtype0o
lstm_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????h
lstm_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: h
lstm_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
lstm_2/strided_slice_3StridedSlice2lstm_2/TensorArrayV2Stack/TensorListStack:tensor:0%lstm_2/strided_slice_3/stack:output:0'lstm_2/strided_slice_3/stack_1:output:0'lstm_2/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_maskl
lstm_2/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
lstm_2/transpose_1	Transpose2lstm_2/TensorArrayV2Stack/TensorListStack:tensor:0 lstm_2/transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????R
lstm_3/ShapeShapelstm_2/transpose_1:y:0*
T0*
_output_shapes
:d
lstm_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: f
lstm_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:f
lstm_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
lstm_3/strided_sliceStridedSlicelstm_3/Shape:output:0#lstm_3/strided_slice/stack:output:0%lstm_3/strided_slice/stack_1:output:0%lstm_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
lstm_3/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :??
lstm_3/zeros/packedPacklstm_3/strided_slice:output:0lstm_3/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:W
lstm_3/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
lstm_3/zerosFilllstm_3/zeros/packed:output:0lstm_3/zeros/Const:output:0*
T0*(
_output_shapes
:??????????Z
lstm_3/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :??
lstm_3/zeros_1/packedPacklstm_3/strided_slice:output:0 lstm_3/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:Y
lstm_3/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
lstm_3/zeros_1Filllstm_3/zeros_1/packed:output:0lstm_3/zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????j
lstm_3/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
lstm_3/transpose	Transposelstm_2/transpose_1:y:0lstm_3/transpose/perm:output:0*
T0*,
_output_shapes
:??????????R
lstm_3/Shape_1Shapelstm_3/transpose:y:0*
T0*
_output_shapes
:f
lstm_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: h
lstm_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
lstm_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
lstm_3/strided_slice_1StridedSlicelstm_3/Shape_1:output:0%lstm_3/strided_slice_1/stack:output:0'lstm_3/strided_slice_1/stack_1:output:0'lstm_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskm
"lstm_3/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
lstm_3/TensorArrayV2TensorListReserve+lstm_3/TensorArrayV2/element_shape:output:0lstm_3/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
<lstm_3/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
.lstm_3/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_3/transpose:y:0Elstm_3/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???f
lstm_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: h
lstm_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
lstm_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
lstm_3/strided_slice_2StridedSlicelstm_3/transpose:y:0%lstm_3/strided_slice_2/stack:output:0'lstm_3/strided_slice_2/stack_1:output:0'lstm_3/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_maskd
"lstm_3/lstm_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
'lstm_3/lstm_cell_3/split/ReadVariableOpReadVariableOp0lstm_3_lstm_cell_3_split_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
lstm_3/lstm_cell_3/splitSplit+lstm_3/lstm_cell_3/split/split_dim:output:0/lstm_3/lstm_cell_3/split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
??:
??:
??:
??*
	num_split?
lstm_3/lstm_cell_3/MatMulMatMullstm_3/strided_slice_2:output:0!lstm_3/lstm_cell_3/split:output:0*
T0*(
_output_shapes
:???????????
lstm_3/lstm_cell_3/MatMul_1MatMullstm_3/strided_slice_2:output:0!lstm_3/lstm_cell_3/split:output:1*
T0*(
_output_shapes
:???????????
lstm_3/lstm_cell_3/MatMul_2MatMullstm_3/strided_slice_2:output:0!lstm_3/lstm_cell_3/split:output:2*
T0*(
_output_shapes
:???????????
lstm_3/lstm_cell_3/MatMul_3MatMullstm_3/strided_slice_2:output:0!lstm_3/lstm_cell_3/split:output:3*
T0*(
_output_shapes
:??????????f
$lstm_3/lstm_cell_3/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
)lstm_3/lstm_cell_3/split_1/ReadVariableOpReadVariableOp2lstm_3_lstm_cell_3_split_1_readvariableop_resource*
_output_shapes	
:?*
dtype0?
lstm_3/lstm_cell_3/split_1Split-lstm_3/lstm_cell_3/split_1/split_dim:output:01lstm_3/lstm_cell_3/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split?
lstm_3/lstm_cell_3/BiasAddBiasAdd#lstm_3/lstm_cell_3/MatMul:product:0#lstm_3/lstm_cell_3/split_1:output:0*
T0*(
_output_shapes
:???????????
lstm_3/lstm_cell_3/BiasAdd_1BiasAdd%lstm_3/lstm_cell_3/MatMul_1:product:0#lstm_3/lstm_cell_3/split_1:output:1*
T0*(
_output_shapes
:???????????
lstm_3/lstm_cell_3/BiasAdd_2BiasAdd%lstm_3/lstm_cell_3/MatMul_2:product:0#lstm_3/lstm_cell_3/split_1:output:2*
T0*(
_output_shapes
:???????????
lstm_3/lstm_cell_3/BiasAdd_3BiasAdd%lstm_3/lstm_cell_3/MatMul_3:product:0#lstm_3/lstm_cell_3/split_1:output:3*
T0*(
_output_shapes
:???????????
!lstm_3/lstm_cell_3/ReadVariableOpReadVariableOp*lstm_3_lstm_cell_3_readvariableop_resource* 
_output_shapes
:
??*
dtype0w
&lstm_3/lstm_cell_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        y
(lstm_3/lstm_cell_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   y
(lstm_3/lstm_cell_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
 lstm_3/lstm_cell_3/strided_sliceStridedSlice)lstm_3/lstm_cell_3/ReadVariableOp:value:0/lstm_3/lstm_cell_3/strided_slice/stack:output:01lstm_3/lstm_cell_3/strided_slice/stack_1:output:01lstm_3/lstm_cell_3/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
lstm_3/lstm_cell_3/MatMul_4MatMullstm_3/zeros:output:0)lstm_3/lstm_cell_3/strided_slice:output:0*
T0*(
_output_shapes
:???????????
lstm_3/lstm_cell_3/addAddV2#lstm_3/lstm_cell_3/BiasAdd:output:0%lstm_3/lstm_cell_3/MatMul_4:product:0*
T0*(
_output_shapes
:??????????]
lstm_3/lstm_cell_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>_
lstm_3/lstm_cell_3/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ??
lstm_3/lstm_cell_3/MulMullstm_3/lstm_cell_3/add:z:0!lstm_3/lstm_cell_3/Const:output:0*
T0*(
_output_shapes
:???????????
lstm_3/lstm_cell_3/Add_1AddV2lstm_3/lstm_cell_3/Mul:z:0#lstm_3/lstm_cell_3/Const_1:output:0*
T0*(
_output_shapes
:??????????o
*lstm_3/lstm_cell_3/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
(lstm_3/lstm_cell_3/clip_by_value/MinimumMinimumlstm_3/lstm_cell_3/Add_1:z:03lstm_3/lstm_cell_3/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:??????????g
"lstm_3/lstm_cell_3/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
 lstm_3/lstm_cell_3/clip_by_valueMaximum,lstm_3/lstm_cell_3/clip_by_value/Minimum:z:0+lstm_3/lstm_cell_3/clip_by_value/y:output:0*
T0*(
_output_shapes
:???????????
#lstm_3/lstm_cell_3/ReadVariableOp_1ReadVariableOp*lstm_3_lstm_cell_3_readvariableop_resource* 
_output_shapes
:
??*
dtype0y
(lstm_3/lstm_cell_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   {
*lstm_3/lstm_cell_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       {
*lstm_3/lstm_cell_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
"lstm_3/lstm_cell_3/strided_slice_1StridedSlice+lstm_3/lstm_cell_3/ReadVariableOp_1:value:01lstm_3/lstm_cell_3/strided_slice_1/stack:output:03lstm_3/lstm_cell_3/strided_slice_1/stack_1:output:03lstm_3/lstm_cell_3/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
lstm_3/lstm_cell_3/MatMul_5MatMullstm_3/zeros:output:0+lstm_3/lstm_cell_3/strided_slice_1:output:0*
T0*(
_output_shapes
:???????????
lstm_3/lstm_cell_3/add_2AddV2%lstm_3/lstm_cell_3/BiasAdd_1:output:0%lstm_3/lstm_cell_3/MatMul_5:product:0*
T0*(
_output_shapes
:??????????_
lstm_3/lstm_cell_3/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>_
lstm_3/lstm_cell_3/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ??
lstm_3/lstm_cell_3/Mul_1Mullstm_3/lstm_cell_3/add_2:z:0#lstm_3/lstm_cell_3/Const_2:output:0*
T0*(
_output_shapes
:???????????
lstm_3/lstm_cell_3/Add_3AddV2lstm_3/lstm_cell_3/Mul_1:z:0#lstm_3/lstm_cell_3/Const_3:output:0*
T0*(
_output_shapes
:??????????q
,lstm_3/lstm_cell_3/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
*lstm_3/lstm_cell_3/clip_by_value_1/MinimumMinimumlstm_3/lstm_cell_3/Add_3:z:05lstm_3/lstm_cell_3/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:??????????i
$lstm_3/lstm_cell_3/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
"lstm_3/lstm_cell_3/clip_by_value_1Maximum.lstm_3/lstm_cell_3/clip_by_value_1/Minimum:z:0-lstm_3/lstm_cell_3/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:???????????
lstm_3/lstm_cell_3/mul_2Mul&lstm_3/lstm_cell_3/clip_by_value_1:z:0lstm_3/zeros_1:output:0*
T0*(
_output_shapes
:???????????
#lstm_3/lstm_cell_3/ReadVariableOp_2ReadVariableOp*lstm_3_lstm_cell_3_readvariableop_resource* 
_output_shapes
:
??*
dtype0y
(lstm_3/lstm_cell_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       {
*lstm_3/lstm_cell_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  {
*lstm_3/lstm_cell_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
"lstm_3/lstm_cell_3/strided_slice_2StridedSlice+lstm_3/lstm_cell_3/ReadVariableOp_2:value:01lstm_3/lstm_cell_3/strided_slice_2/stack:output:03lstm_3/lstm_cell_3/strided_slice_2/stack_1:output:03lstm_3/lstm_cell_3/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
lstm_3/lstm_cell_3/MatMul_6MatMullstm_3/zeros:output:0+lstm_3/lstm_cell_3/strided_slice_2:output:0*
T0*(
_output_shapes
:???????????
lstm_3/lstm_cell_3/add_4AddV2%lstm_3/lstm_cell_3/BiasAdd_2:output:0%lstm_3/lstm_cell_3/MatMul_6:product:0*
T0*(
_output_shapes
:??????????p
lstm_3/lstm_cell_3/TanhTanhlstm_3/lstm_cell_3/add_4:z:0*
T0*(
_output_shapes
:???????????
lstm_3/lstm_cell_3/mul_3Mul$lstm_3/lstm_cell_3/clip_by_value:z:0lstm_3/lstm_cell_3/Tanh:y:0*
T0*(
_output_shapes
:???????????
lstm_3/lstm_cell_3/add_5AddV2lstm_3/lstm_cell_3/mul_2:z:0lstm_3/lstm_cell_3/mul_3:z:0*
T0*(
_output_shapes
:???????????
#lstm_3/lstm_cell_3/ReadVariableOp_3ReadVariableOp*lstm_3_lstm_cell_3_readvariableop_resource* 
_output_shapes
:
??*
dtype0y
(lstm_3/lstm_cell_3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  {
*lstm_3/lstm_cell_3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        {
*lstm_3/lstm_cell_3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
"lstm_3/lstm_cell_3/strided_slice_3StridedSlice+lstm_3/lstm_cell_3/ReadVariableOp_3:value:01lstm_3/lstm_cell_3/strided_slice_3/stack:output:03lstm_3/lstm_cell_3/strided_slice_3/stack_1:output:03lstm_3/lstm_cell_3/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
lstm_3/lstm_cell_3/MatMul_7MatMullstm_3/zeros:output:0+lstm_3/lstm_cell_3/strided_slice_3:output:0*
T0*(
_output_shapes
:???????????
lstm_3/lstm_cell_3/add_6AddV2%lstm_3/lstm_cell_3/BiasAdd_3:output:0%lstm_3/lstm_cell_3/MatMul_7:product:0*
T0*(
_output_shapes
:??????????_
lstm_3/lstm_cell_3/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>_
lstm_3/lstm_cell_3/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ??
lstm_3/lstm_cell_3/Mul_4Mullstm_3/lstm_cell_3/add_6:z:0#lstm_3/lstm_cell_3/Const_4:output:0*
T0*(
_output_shapes
:???????????
lstm_3/lstm_cell_3/Add_7AddV2lstm_3/lstm_cell_3/Mul_4:z:0#lstm_3/lstm_cell_3/Const_5:output:0*
T0*(
_output_shapes
:??????????q
,lstm_3/lstm_cell_3/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
*lstm_3/lstm_cell_3/clip_by_value_2/MinimumMinimumlstm_3/lstm_cell_3/Add_7:z:05lstm_3/lstm_cell_3/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:??????????i
$lstm_3/lstm_cell_3/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
"lstm_3/lstm_cell_3/clip_by_value_2Maximum.lstm_3/lstm_cell_3/clip_by_value_2/Minimum:z:0-lstm_3/lstm_cell_3/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:??????????r
lstm_3/lstm_cell_3/Tanh_1Tanhlstm_3/lstm_cell_3/add_5:z:0*
T0*(
_output_shapes
:???????????
lstm_3/lstm_cell_3/mul_5Mul&lstm_3/lstm_cell_3/clip_by_value_2:z:0lstm_3/lstm_cell_3/Tanh_1:y:0*
T0*(
_output_shapes
:??????????u
$lstm_3/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   ?
lstm_3/TensorArrayV2_1TensorListReserve-lstm_3/TensorArrayV2_1/element_shape:output:0lstm_3/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???M
lstm_3/timeConst*
_output_shapes
: *
dtype0*
value	B : j
lstm_3/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????[
lstm_3/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
lstm_3/whileWhile"lstm_3/while/loop_counter:output:0(lstm_3/while/maximum_iterations:output:0lstm_3/time:output:0lstm_3/TensorArrayV2_1:handle:0lstm_3/zeros:output:0lstm_3/zeros_1:output:0lstm_3/strided_slice_1:output:0>lstm_3/TensorArrayUnstack/TensorListFromTensor:output_handle:00lstm_3_lstm_cell_3_split_readvariableop_resource2lstm_3_lstm_cell_3_split_1_readvariableop_resource*lstm_3_lstm_cell_3_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *#
bodyR
lstm_3_while_body_57117*#
condR
lstm_3_while_cond_57116*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations ?
7lstm_3/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   ?
)lstm_3/TensorArrayV2Stack/TensorListStackTensorListStacklstm_3/while:output:3@lstm_3/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
element_dtype0o
lstm_3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????h
lstm_3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: h
lstm_3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
lstm_3/strided_slice_3StridedSlice2lstm_3/TensorArrayV2Stack/TensorListStack:tensor:0%lstm_3/strided_slice_3/stack:output:0'lstm_3/strided_slice_3/stack_1:output:0'lstm_3/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_maskl
lstm_3/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
lstm_3/transpose_1	Transpose2lstm_3/TensorArrayV2Stack/TensorListStack:tensor:0 lstm_3/transpose_1/perm:output:0*
T0*,
_output_shapes
:???????????
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
dense_2/MatMulMatMullstm_3/strided_slice_3:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????a
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
dense_3/MatMulMatMuldense_2/Relu:activations:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????f
dense_3/SigmoidSigmoiddense_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????b
IdentityIdentitydense_3/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp"^lstm_2/lstm_cell_2/ReadVariableOp$^lstm_2/lstm_cell_2/ReadVariableOp_1$^lstm_2/lstm_cell_2/ReadVariableOp_2$^lstm_2/lstm_cell_2/ReadVariableOp_3(^lstm_2/lstm_cell_2/split/ReadVariableOp*^lstm_2/lstm_cell_2/split_1/ReadVariableOp^lstm_2/while"^lstm_3/lstm_cell_3/ReadVariableOp$^lstm_3/lstm_cell_3/ReadVariableOp_1$^lstm_3/lstm_cell_3/ReadVariableOp_2$^lstm_3/lstm_cell_3/ReadVariableOp_3(^lstm_3/lstm_cell_3/split/ReadVariableOp*^lstm_3/lstm_cell_3/split_1/ReadVariableOp^lstm_3/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????: : : : : : : : : : 2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2F
!lstm_2/lstm_cell_2/ReadVariableOp!lstm_2/lstm_cell_2/ReadVariableOp2J
#lstm_2/lstm_cell_2/ReadVariableOp_1#lstm_2/lstm_cell_2/ReadVariableOp_12J
#lstm_2/lstm_cell_2/ReadVariableOp_2#lstm_2/lstm_cell_2/ReadVariableOp_22J
#lstm_2/lstm_cell_2/ReadVariableOp_3#lstm_2/lstm_cell_2/ReadVariableOp_32R
'lstm_2/lstm_cell_2/split/ReadVariableOp'lstm_2/lstm_cell_2/split/ReadVariableOp2V
)lstm_2/lstm_cell_2/split_1/ReadVariableOp)lstm_2/lstm_cell_2/split_1/ReadVariableOp2
lstm_2/whilelstm_2/while2F
!lstm_3/lstm_cell_3/ReadVariableOp!lstm_3/lstm_cell_3/ReadVariableOp2J
#lstm_3/lstm_cell_3/ReadVariableOp_1#lstm_3/lstm_cell_3/ReadVariableOp_12J
#lstm_3/lstm_cell_3/ReadVariableOp_2#lstm_3/lstm_cell_3/ReadVariableOp_22J
#lstm_3/lstm_cell_3/ReadVariableOp_3#lstm_3/lstm_cell_3/ReadVariableOp_32R
'lstm_3/lstm_cell_3/split/ReadVariableOp'lstm_3/lstm_cell_3/split/ReadVariableOp2V
)lstm_3/lstm_cell_3/split_1/ReadVariableOp)lstm_3/lstm_cell_3/split_1/ReadVariableOp2
lstm_3/whilelstm_3/while:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?
A__inference_lstm_3_layer_call_and_return_conditional_losses_55748

inputs=
)lstm_cell_3_split_readvariableop_resource:
??:
+lstm_cell_3_split_1_readvariableop_resource:	?7
#lstm_cell_3_readvariableop_resource:
??
identity??lstm_cell_3/ReadVariableOp?lstm_cell_3/ReadVariableOp_1?lstm_cell_3/ReadVariableOp_2?lstm_cell_3/ReadVariableOp_3? lstm_cell_3/split/ReadVariableOp?"lstm_cell_3/split_1/ReadVariableOp?while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????S
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:??????????D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask]
lstm_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
 lstm_cell_3/split/ReadVariableOpReadVariableOp)lstm_cell_3_split_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
lstm_cell_3/splitSplit$lstm_cell_3/split/split_dim:output:0(lstm_cell_3/split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
??:
??:
??:
??*
	num_split?
lstm_cell_3/MatMulMatMulstrided_slice_2:output:0lstm_cell_3/split:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_3/MatMul_1MatMulstrided_slice_2:output:0lstm_cell_3/split:output:1*
T0*(
_output_shapes
:???????????
lstm_cell_3/MatMul_2MatMulstrided_slice_2:output:0lstm_cell_3/split:output:2*
T0*(
_output_shapes
:???????????
lstm_cell_3/MatMul_3MatMulstrided_slice_2:output:0lstm_cell_3/split:output:3*
T0*(
_output_shapes
:??????????_
lstm_cell_3/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
"lstm_cell_3/split_1/ReadVariableOpReadVariableOp+lstm_cell_3_split_1_readvariableop_resource*
_output_shapes	
:?*
dtype0?
lstm_cell_3/split_1Split&lstm_cell_3/split_1/split_dim:output:0*lstm_cell_3/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split?
lstm_cell_3/BiasAddBiasAddlstm_cell_3/MatMul:product:0lstm_cell_3/split_1:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_3/BiasAdd_1BiasAddlstm_cell_3/MatMul_1:product:0lstm_cell_3/split_1:output:1*
T0*(
_output_shapes
:???????????
lstm_cell_3/BiasAdd_2BiasAddlstm_cell_3/MatMul_2:product:0lstm_cell_3/split_1:output:2*
T0*(
_output_shapes
:???????????
lstm_cell_3/BiasAdd_3BiasAddlstm_cell_3/MatMul_3:product:0lstm_cell_3/split_1:output:3*
T0*(
_output_shapes
:???????????
lstm_cell_3/ReadVariableOpReadVariableOp#lstm_cell_3_readvariableop_resource* 
_output_shapes
:
??*
dtype0p
lstm_cell_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        r
!lstm_cell_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   r
!lstm_cell_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
lstm_cell_3/strided_sliceStridedSlice"lstm_cell_3/ReadVariableOp:value:0(lstm_cell_3/strided_slice/stack:output:0*lstm_cell_3/strided_slice/stack_1:output:0*lstm_cell_3/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
lstm_cell_3/MatMul_4MatMulzeros:output:0"lstm_cell_3/strided_slice:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_3/addAddV2lstm_cell_3/BiasAdd:output:0lstm_cell_3/MatMul_4:product:0*
T0*(
_output_shapes
:??????????V
lstm_cell_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>X
lstm_cell_3/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?z
lstm_cell_3/MulMullstm_cell_3/add:z:0lstm_cell_3/Const:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_3/Add_1AddV2lstm_cell_3/Mul:z:0lstm_cell_3/Const_1:output:0*
T0*(
_output_shapes
:??????????h
#lstm_cell_3/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
!lstm_cell_3/clip_by_value/MinimumMinimumlstm_cell_3/Add_1:z:0,lstm_cell_3/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:??????????`
lstm_cell_3/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
lstm_cell_3/clip_by_valueMaximum%lstm_cell_3/clip_by_value/Minimum:z:0$lstm_cell_3/clip_by_value/y:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_3/ReadVariableOp_1ReadVariableOp#lstm_cell_3_readvariableop_resource* 
_output_shapes
:
??*
dtype0r
!lstm_cell_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   t
#lstm_cell_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       t
#lstm_cell_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
lstm_cell_3/strided_slice_1StridedSlice$lstm_cell_3/ReadVariableOp_1:value:0*lstm_cell_3/strided_slice_1/stack:output:0,lstm_cell_3/strided_slice_1/stack_1:output:0,lstm_cell_3/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
lstm_cell_3/MatMul_5MatMulzeros:output:0$lstm_cell_3/strided_slice_1:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_3/add_2AddV2lstm_cell_3/BiasAdd_1:output:0lstm_cell_3/MatMul_5:product:0*
T0*(
_output_shapes
:??????????X
lstm_cell_3/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>X
lstm_cell_3/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ??
lstm_cell_3/Mul_1Mullstm_cell_3/add_2:z:0lstm_cell_3/Const_2:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_3/Add_3AddV2lstm_cell_3/Mul_1:z:0lstm_cell_3/Const_3:output:0*
T0*(
_output_shapes
:??????????j
%lstm_cell_3/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
#lstm_cell_3/clip_by_value_1/MinimumMinimumlstm_cell_3/Add_3:z:0.lstm_cell_3/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:??????????b
lstm_cell_3/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
lstm_cell_3/clip_by_value_1Maximum'lstm_cell_3/clip_by_value_1/Minimum:z:0&lstm_cell_3/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:??????????~
lstm_cell_3/mul_2Mullstm_cell_3/clip_by_value_1:z:0zeros_1:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_3/ReadVariableOp_2ReadVariableOp#lstm_cell_3_readvariableop_resource* 
_output_shapes
:
??*
dtype0r
!lstm_cell_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       t
#lstm_cell_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  t
#lstm_cell_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
lstm_cell_3/strided_slice_2StridedSlice$lstm_cell_3/ReadVariableOp_2:value:0*lstm_cell_3/strided_slice_2/stack:output:0,lstm_cell_3/strided_slice_2/stack_1:output:0,lstm_cell_3/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
lstm_cell_3/MatMul_6MatMulzeros:output:0$lstm_cell_3/strided_slice_2:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_3/add_4AddV2lstm_cell_3/BiasAdd_2:output:0lstm_cell_3/MatMul_6:product:0*
T0*(
_output_shapes
:??????????b
lstm_cell_3/TanhTanhlstm_cell_3/add_4:z:0*
T0*(
_output_shapes
:???????????
lstm_cell_3/mul_3Mullstm_cell_3/clip_by_value:z:0lstm_cell_3/Tanh:y:0*
T0*(
_output_shapes
:??????????{
lstm_cell_3/add_5AddV2lstm_cell_3/mul_2:z:0lstm_cell_3/mul_3:z:0*
T0*(
_output_shapes
:???????????
lstm_cell_3/ReadVariableOp_3ReadVariableOp#lstm_cell_3_readvariableop_resource* 
_output_shapes
:
??*
dtype0r
!lstm_cell_3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  t
#lstm_cell_3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        t
#lstm_cell_3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
lstm_cell_3/strided_slice_3StridedSlice$lstm_cell_3/ReadVariableOp_3:value:0*lstm_cell_3/strided_slice_3/stack:output:0,lstm_cell_3/strided_slice_3/stack_1:output:0,lstm_cell_3/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
lstm_cell_3/MatMul_7MatMulzeros:output:0$lstm_cell_3/strided_slice_3:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_3/add_6AddV2lstm_cell_3/BiasAdd_3:output:0lstm_cell_3/MatMul_7:product:0*
T0*(
_output_shapes
:??????????X
lstm_cell_3/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>X
lstm_cell_3/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ??
lstm_cell_3/Mul_4Mullstm_cell_3/add_6:z:0lstm_cell_3/Const_4:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_3/Add_7AddV2lstm_cell_3/Mul_4:z:0lstm_cell_3/Const_5:output:0*
T0*(
_output_shapes
:??????????j
%lstm_cell_3/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
#lstm_cell_3/clip_by_value_2/MinimumMinimumlstm_cell_3/Add_7:z:0.lstm_cell_3/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:??????????b
lstm_cell_3/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
lstm_cell_3/clip_by_value_2Maximum'lstm_cell_3/clip_by_value_2/Minimum:z:0&lstm_cell_3/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:??????????d
lstm_cell_3/Tanh_1Tanhlstm_cell_3/add_5:z:0*
T0*(
_output_shapes
:???????????
lstm_cell_3/mul_5Mullstm_cell_3/clip_by_value_2:z:0lstm_cell_3/Tanh_1:y:0*
T0*(
_output_shapes
:??????????n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   ?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)lstm_cell_3_split_readvariableop_resource+lstm_cell_3_split_1_readvariableop_resource#lstm_cell_3_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_55608*
condR
while_cond_55607*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????h
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp^lstm_cell_3/ReadVariableOp^lstm_cell_3/ReadVariableOp_1^lstm_cell_3/ReadVariableOp_2^lstm_cell_3/ReadVariableOp_3!^lstm_cell_3/split/ReadVariableOp#^lstm_cell_3/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:??????????: : : 28
lstm_cell_3/ReadVariableOplstm_cell_3/ReadVariableOp2<
lstm_cell_3/ReadVariableOp_1lstm_cell_3/ReadVariableOp_12<
lstm_cell_3/ReadVariableOp_2lstm_cell_3/ReadVariableOp_22<
lstm_cell_3/ReadVariableOp_3lstm_cell_3/ReadVariableOp_32D
 lstm_cell_3/split/ReadVariableOp lstm_cell_3/split/ReadVariableOp2H
"lstm_cell_3/split_1/ReadVariableOp"lstm_cell_3/split_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
while_cond_55885
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_55885___redundant_placeholder03
/while_while_cond_55885___redundant_placeholder13
/while_while_cond_55885___redundant_placeholder23
/while_while_cond_55885___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
??
?
$sequential_1_lstm_2_while_body_53538D
@sequential_1_lstm_2_while_sequential_1_lstm_2_while_loop_counterJ
Fsequential_1_lstm_2_while_sequential_1_lstm_2_while_maximum_iterations)
%sequential_1_lstm_2_while_placeholder+
'sequential_1_lstm_2_while_placeholder_1+
'sequential_1_lstm_2_while_placeholder_2+
'sequential_1_lstm_2_while_placeholder_3C
?sequential_1_lstm_2_while_sequential_1_lstm_2_strided_slice_1_0
{sequential_1_lstm_2_while_tensorarrayv2read_tensorlistgetitem_sequential_1_lstm_2_tensorarrayunstack_tensorlistfromtensor_0X
Esequential_1_lstm_2_while_lstm_cell_2_split_readvariableop_resource_0:	?V
Gsequential_1_lstm_2_while_lstm_cell_2_split_1_readvariableop_resource_0:	?S
?sequential_1_lstm_2_while_lstm_cell_2_readvariableop_resource_0:
??&
"sequential_1_lstm_2_while_identity(
$sequential_1_lstm_2_while_identity_1(
$sequential_1_lstm_2_while_identity_2(
$sequential_1_lstm_2_while_identity_3(
$sequential_1_lstm_2_while_identity_4(
$sequential_1_lstm_2_while_identity_5A
=sequential_1_lstm_2_while_sequential_1_lstm_2_strided_slice_1}
ysequential_1_lstm_2_while_tensorarrayv2read_tensorlistgetitem_sequential_1_lstm_2_tensorarrayunstack_tensorlistfromtensorV
Csequential_1_lstm_2_while_lstm_cell_2_split_readvariableop_resource:	?T
Esequential_1_lstm_2_while_lstm_cell_2_split_1_readvariableop_resource:	?Q
=sequential_1_lstm_2_while_lstm_cell_2_readvariableop_resource:
????4sequential_1/lstm_2/while/lstm_cell_2/ReadVariableOp?6sequential_1/lstm_2/while/lstm_cell_2/ReadVariableOp_1?6sequential_1/lstm_2/while/lstm_cell_2/ReadVariableOp_2?6sequential_1/lstm_2/while/lstm_cell_2/ReadVariableOp_3?:sequential_1/lstm_2/while/lstm_cell_2/split/ReadVariableOp?<sequential_1/lstm_2/while/lstm_cell_2/split_1/ReadVariableOp?
Ksequential_1/lstm_2/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
=sequential_1/lstm_2/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem{sequential_1_lstm_2_while_tensorarrayv2read_tensorlistgetitem_sequential_1_lstm_2_tensorarrayunstack_tensorlistfromtensor_0%sequential_1_lstm_2_while_placeholderTsequential_1/lstm_2/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0w
5sequential_1/lstm_2/while/lstm_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
:sequential_1/lstm_2/while/lstm_cell_2/split/ReadVariableOpReadVariableOpEsequential_1_lstm_2_while_lstm_cell_2_split_readvariableop_resource_0*
_output_shapes
:	?*
dtype0?
+sequential_1/lstm_2/while/lstm_cell_2/splitSplit>sequential_1/lstm_2/while/lstm_cell_2/split/split_dim:output:0Bsequential_1/lstm_2/while/lstm_cell_2/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	?:	?:	?:	?*
	num_split?
,sequential_1/lstm_2/while/lstm_cell_2/MatMulMatMulDsequential_1/lstm_2/while/TensorArrayV2Read/TensorListGetItem:item:04sequential_1/lstm_2/while/lstm_cell_2/split:output:0*
T0*(
_output_shapes
:???????????
.sequential_1/lstm_2/while/lstm_cell_2/MatMul_1MatMulDsequential_1/lstm_2/while/TensorArrayV2Read/TensorListGetItem:item:04sequential_1/lstm_2/while/lstm_cell_2/split:output:1*
T0*(
_output_shapes
:???????????
.sequential_1/lstm_2/while/lstm_cell_2/MatMul_2MatMulDsequential_1/lstm_2/while/TensorArrayV2Read/TensorListGetItem:item:04sequential_1/lstm_2/while/lstm_cell_2/split:output:2*
T0*(
_output_shapes
:???????????
.sequential_1/lstm_2/while/lstm_cell_2/MatMul_3MatMulDsequential_1/lstm_2/while/TensorArrayV2Read/TensorListGetItem:item:04sequential_1/lstm_2/while/lstm_cell_2/split:output:3*
T0*(
_output_shapes
:??????????y
7sequential_1/lstm_2/while/lstm_cell_2/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
<sequential_1/lstm_2/while/lstm_cell_2/split_1/ReadVariableOpReadVariableOpGsequential_1_lstm_2_while_lstm_cell_2_split_1_readvariableop_resource_0*
_output_shapes	
:?*
dtype0?
-sequential_1/lstm_2/while/lstm_cell_2/split_1Split@sequential_1/lstm_2/while/lstm_cell_2/split_1/split_dim:output:0Dsequential_1/lstm_2/while/lstm_cell_2/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split?
-sequential_1/lstm_2/while/lstm_cell_2/BiasAddBiasAdd6sequential_1/lstm_2/while/lstm_cell_2/MatMul:product:06sequential_1/lstm_2/while/lstm_cell_2/split_1:output:0*
T0*(
_output_shapes
:???????????
/sequential_1/lstm_2/while/lstm_cell_2/BiasAdd_1BiasAdd8sequential_1/lstm_2/while/lstm_cell_2/MatMul_1:product:06sequential_1/lstm_2/while/lstm_cell_2/split_1:output:1*
T0*(
_output_shapes
:???????????
/sequential_1/lstm_2/while/lstm_cell_2/BiasAdd_2BiasAdd8sequential_1/lstm_2/while/lstm_cell_2/MatMul_2:product:06sequential_1/lstm_2/while/lstm_cell_2/split_1:output:2*
T0*(
_output_shapes
:???????????
/sequential_1/lstm_2/while/lstm_cell_2/BiasAdd_3BiasAdd8sequential_1/lstm_2/while/lstm_cell_2/MatMul_3:product:06sequential_1/lstm_2/while/lstm_cell_2/split_1:output:3*
T0*(
_output_shapes
:???????????
4sequential_1/lstm_2/while/lstm_cell_2/ReadVariableOpReadVariableOp?sequential_1_lstm_2_while_lstm_cell_2_readvariableop_resource_0* 
_output_shapes
:
??*
dtype0?
9sequential_1/lstm_2/while/lstm_cell_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
;sequential_1/lstm_2/while/lstm_cell_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
;sequential_1/lstm_2/while/lstm_cell_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
3sequential_1/lstm_2/while/lstm_cell_2/strided_sliceStridedSlice<sequential_1/lstm_2/while/lstm_cell_2/ReadVariableOp:value:0Bsequential_1/lstm_2/while/lstm_cell_2/strided_slice/stack:output:0Dsequential_1/lstm_2/while/lstm_cell_2/strided_slice/stack_1:output:0Dsequential_1/lstm_2/while/lstm_cell_2/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
.sequential_1/lstm_2/while/lstm_cell_2/MatMul_4MatMul'sequential_1_lstm_2_while_placeholder_2<sequential_1/lstm_2/while/lstm_cell_2/strided_slice:output:0*
T0*(
_output_shapes
:???????????
)sequential_1/lstm_2/while/lstm_cell_2/addAddV26sequential_1/lstm_2/while/lstm_cell_2/BiasAdd:output:08sequential_1/lstm_2/while/lstm_cell_2/MatMul_4:product:0*
T0*(
_output_shapes
:??????????p
+sequential_1/lstm_2/while/lstm_cell_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>r
-sequential_1/lstm_2/while/lstm_cell_2/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ??
)sequential_1/lstm_2/while/lstm_cell_2/MulMul-sequential_1/lstm_2/while/lstm_cell_2/add:z:04sequential_1/lstm_2/while/lstm_cell_2/Const:output:0*
T0*(
_output_shapes
:???????????
+sequential_1/lstm_2/while/lstm_cell_2/Add_1AddV2-sequential_1/lstm_2/while/lstm_cell_2/Mul:z:06sequential_1/lstm_2/while/lstm_cell_2/Const_1:output:0*
T0*(
_output_shapes
:???????????
=sequential_1/lstm_2/while/lstm_cell_2/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
;sequential_1/lstm_2/while/lstm_cell_2/clip_by_value/MinimumMinimum/sequential_1/lstm_2/while/lstm_cell_2/Add_1:z:0Fsequential_1/lstm_2/while/lstm_cell_2/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:??????????z
5sequential_1/lstm_2/while/lstm_cell_2/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
3sequential_1/lstm_2/while/lstm_cell_2/clip_by_valueMaximum?sequential_1/lstm_2/while/lstm_cell_2/clip_by_value/Minimum:z:0>sequential_1/lstm_2/while/lstm_cell_2/clip_by_value/y:output:0*
T0*(
_output_shapes
:???????????
6sequential_1/lstm_2/while/lstm_cell_2/ReadVariableOp_1ReadVariableOp?sequential_1_lstm_2_while_lstm_cell_2_readvariableop_resource_0* 
_output_shapes
:
??*
dtype0?
;sequential_1/lstm_2/while/lstm_cell_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       ?
=sequential_1/lstm_2/while/lstm_cell_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
=sequential_1/lstm_2/while/lstm_cell_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
5sequential_1/lstm_2/while/lstm_cell_2/strided_slice_1StridedSlice>sequential_1/lstm_2/while/lstm_cell_2/ReadVariableOp_1:value:0Dsequential_1/lstm_2/while/lstm_cell_2/strided_slice_1/stack:output:0Fsequential_1/lstm_2/while/lstm_cell_2/strided_slice_1/stack_1:output:0Fsequential_1/lstm_2/while/lstm_cell_2/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
.sequential_1/lstm_2/while/lstm_cell_2/MatMul_5MatMul'sequential_1_lstm_2_while_placeholder_2>sequential_1/lstm_2/while/lstm_cell_2/strided_slice_1:output:0*
T0*(
_output_shapes
:???????????
+sequential_1/lstm_2/while/lstm_cell_2/add_2AddV28sequential_1/lstm_2/while/lstm_cell_2/BiasAdd_1:output:08sequential_1/lstm_2/while/lstm_cell_2/MatMul_5:product:0*
T0*(
_output_shapes
:??????????r
-sequential_1/lstm_2/while/lstm_cell_2/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>r
-sequential_1/lstm_2/while/lstm_cell_2/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ??
+sequential_1/lstm_2/while/lstm_cell_2/Mul_1Mul/sequential_1/lstm_2/while/lstm_cell_2/add_2:z:06sequential_1/lstm_2/while/lstm_cell_2/Const_2:output:0*
T0*(
_output_shapes
:???????????
+sequential_1/lstm_2/while/lstm_cell_2/Add_3AddV2/sequential_1/lstm_2/while/lstm_cell_2/Mul_1:z:06sequential_1/lstm_2/while/lstm_cell_2/Const_3:output:0*
T0*(
_output_shapes
:???????????
?sequential_1/lstm_2/while/lstm_cell_2/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
=sequential_1/lstm_2/while/lstm_cell_2/clip_by_value_1/MinimumMinimum/sequential_1/lstm_2/while/lstm_cell_2/Add_3:z:0Hsequential_1/lstm_2/while/lstm_cell_2/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:??????????|
7sequential_1/lstm_2/while/lstm_cell_2/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
5sequential_1/lstm_2/while/lstm_cell_2/clip_by_value_1MaximumAsequential_1/lstm_2/while/lstm_cell_2/clip_by_value_1/Minimum:z:0@sequential_1/lstm_2/while/lstm_cell_2/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:???????????
+sequential_1/lstm_2/while/lstm_cell_2/mul_2Mul9sequential_1/lstm_2/while/lstm_cell_2/clip_by_value_1:z:0'sequential_1_lstm_2_while_placeholder_3*
T0*(
_output_shapes
:???????????
6sequential_1/lstm_2/while/lstm_cell_2/ReadVariableOp_2ReadVariableOp?sequential_1_lstm_2_while_lstm_cell_2_readvariableop_resource_0* 
_output_shapes
:
??*
dtype0?
;sequential_1/lstm_2/while/lstm_cell_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       ?
=sequential_1/lstm_2/while/lstm_cell_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
=sequential_1/lstm_2/while/lstm_cell_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
5sequential_1/lstm_2/while/lstm_cell_2/strided_slice_2StridedSlice>sequential_1/lstm_2/while/lstm_cell_2/ReadVariableOp_2:value:0Dsequential_1/lstm_2/while/lstm_cell_2/strided_slice_2/stack:output:0Fsequential_1/lstm_2/while/lstm_cell_2/strided_slice_2/stack_1:output:0Fsequential_1/lstm_2/while/lstm_cell_2/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
.sequential_1/lstm_2/while/lstm_cell_2/MatMul_6MatMul'sequential_1_lstm_2_while_placeholder_2>sequential_1/lstm_2/while/lstm_cell_2/strided_slice_2:output:0*
T0*(
_output_shapes
:???????????
+sequential_1/lstm_2/while/lstm_cell_2/add_4AddV28sequential_1/lstm_2/while/lstm_cell_2/BiasAdd_2:output:08sequential_1/lstm_2/while/lstm_cell_2/MatMul_6:product:0*
T0*(
_output_shapes
:???????????
*sequential_1/lstm_2/while/lstm_cell_2/TanhTanh/sequential_1/lstm_2/while/lstm_cell_2/add_4:z:0*
T0*(
_output_shapes
:???????????
+sequential_1/lstm_2/while/lstm_cell_2/mul_3Mul7sequential_1/lstm_2/while/lstm_cell_2/clip_by_value:z:0.sequential_1/lstm_2/while/lstm_cell_2/Tanh:y:0*
T0*(
_output_shapes
:???????????
+sequential_1/lstm_2/while/lstm_cell_2/add_5AddV2/sequential_1/lstm_2/while/lstm_cell_2/mul_2:z:0/sequential_1/lstm_2/while/lstm_cell_2/mul_3:z:0*
T0*(
_output_shapes
:???????????
6sequential_1/lstm_2/while/lstm_cell_2/ReadVariableOp_3ReadVariableOp?sequential_1_lstm_2_while_lstm_cell_2_readvariableop_resource_0* 
_output_shapes
:
??*
dtype0?
;sequential_1/lstm_2/while/lstm_cell_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       ?
=sequential_1/lstm_2/while/lstm_cell_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        ?
=sequential_1/lstm_2/while/lstm_cell_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
5sequential_1/lstm_2/while/lstm_cell_2/strided_slice_3StridedSlice>sequential_1/lstm_2/while/lstm_cell_2/ReadVariableOp_3:value:0Dsequential_1/lstm_2/while/lstm_cell_2/strided_slice_3/stack:output:0Fsequential_1/lstm_2/while/lstm_cell_2/strided_slice_3/stack_1:output:0Fsequential_1/lstm_2/while/lstm_cell_2/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
.sequential_1/lstm_2/while/lstm_cell_2/MatMul_7MatMul'sequential_1_lstm_2_while_placeholder_2>sequential_1/lstm_2/while/lstm_cell_2/strided_slice_3:output:0*
T0*(
_output_shapes
:???????????
+sequential_1/lstm_2/while/lstm_cell_2/add_6AddV28sequential_1/lstm_2/while/lstm_cell_2/BiasAdd_3:output:08sequential_1/lstm_2/while/lstm_cell_2/MatMul_7:product:0*
T0*(
_output_shapes
:??????????r
-sequential_1/lstm_2/while/lstm_cell_2/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>r
-sequential_1/lstm_2/while/lstm_cell_2/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ??
+sequential_1/lstm_2/while/lstm_cell_2/Mul_4Mul/sequential_1/lstm_2/while/lstm_cell_2/add_6:z:06sequential_1/lstm_2/while/lstm_cell_2/Const_4:output:0*
T0*(
_output_shapes
:???????????
+sequential_1/lstm_2/while/lstm_cell_2/Add_7AddV2/sequential_1/lstm_2/while/lstm_cell_2/Mul_4:z:06sequential_1/lstm_2/while/lstm_cell_2/Const_5:output:0*
T0*(
_output_shapes
:???????????
?sequential_1/lstm_2/while/lstm_cell_2/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
=sequential_1/lstm_2/while/lstm_cell_2/clip_by_value_2/MinimumMinimum/sequential_1/lstm_2/while/lstm_cell_2/Add_7:z:0Hsequential_1/lstm_2/while/lstm_cell_2/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:??????????|
7sequential_1/lstm_2/while/lstm_cell_2/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
5sequential_1/lstm_2/while/lstm_cell_2/clip_by_value_2MaximumAsequential_1/lstm_2/while/lstm_cell_2/clip_by_value_2/Minimum:z:0@sequential_1/lstm_2/while/lstm_cell_2/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:???????????
,sequential_1/lstm_2/while/lstm_cell_2/Tanh_1Tanh/sequential_1/lstm_2/while/lstm_cell_2/add_5:z:0*
T0*(
_output_shapes
:???????????
+sequential_1/lstm_2/while/lstm_cell_2/mul_5Mul9sequential_1/lstm_2/while/lstm_cell_2/clip_by_value_2:z:00sequential_1/lstm_2/while/lstm_cell_2/Tanh_1:y:0*
T0*(
_output_shapes
:???????????
>sequential_1/lstm_2/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem'sequential_1_lstm_2_while_placeholder_1%sequential_1_lstm_2_while_placeholder/sequential_1/lstm_2/while/lstm_cell_2/mul_5:z:0*
_output_shapes
: *
element_dtype0:???a
sequential_1/lstm_2/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
sequential_1/lstm_2/while/addAddV2%sequential_1_lstm_2_while_placeholder(sequential_1/lstm_2/while/add/y:output:0*
T0*
_output_shapes
: c
!sequential_1/lstm_2/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :?
sequential_1/lstm_2/while/add_1AddV2@sequential_1_lstm_2_while_sequential_1_lstm_2_while_loop_counter*sequential_1/lstm_2/while/add_1/y:output:0*
T0*
_output_shapes
: ?
"sequential_1/lstm_2/while/IdentityIdentity#sequential_1/lstm_2/while/add_1:z:0^sequential_1/lstm_2/while/NoOp*
T0*
_output_shapes
: ?
$sequential_1/lstm_2/while/Identity_1IdentityFsequential_1_lstm_2_while_sequential_1_lstm_2_while_maximum_iterations^sequential_1/lstm_2/while/NoOp*
T0*
_output_shapes
: ?
$sequential_1/lstm_2/while/Identity_2Identity!sequential_1/lstm_2/while/add:z:0^sequential_1/lstm_2/while/NoOp*
T0*
_output_shapes
: ?
$sequential_1/lstm_2/while/Identity_3IdentityNsequential_1/lstm_2/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^sequential_1/lstm_2/while/NoOp*
T0*
_output_shapes
: ?
$sequential_1/lstm_2/while/Identity_4Identity/sequential_1/lstm_2/while/lstm_cell_2/mul_5:z:0^sequential_1/lstm_2/while/NoOp*
T0*(
_output_shapes
:???????????
$sequential_1/lstm_2/while/Identity_5Identity/sequential_1/lstm_2/while/lstm_cell_2/add_5:z:0^sequential_1/lstm_2/while/NoOp*
T0*(
_output_shapes
:???????????
sequential_1/lstm_2/while/NoOpNoOp5^sequential_1/lstm_2/while/lstm_cell_2/ReadVariableOp7^sequential_1/lstm_2/while/lstm_cell_2/ReadVariableOp_17^sequential_1/lstm_2/while/lstm_cell_2/ReadVariableOp_27^sequential_1/lstm_2/while/lstm_cell_2/ReadVariableOp_3;^sequential_1/lstm_2/while/lstm_cell_2/split/ReadVariableOp=^sequential_1/lstm_2/while/lstm_cell_2/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "Q
"sequential_1_lstm_2_while_identity+sequential_1/lstm_2/while/Identity:output:0"U
$sequential_1_lstm_2_while_identity_1-sequential_1/lstm_2/while/Identity_1:output:0"U
$sequential_1_lstm_2_while_identity_2-sequential_1/lstm_2/while/Identity_2:output:0"U
$sequential_1_lstm_2_while_identity_3-sequential_1/lstm_2/while/Identity_3:output:0"U
$sequential_1_lstm_2_while_identity_4-sequential_1/lstm_2/while/Identity_4:output:0"U
$sequential_1_lstm_2_while_identity_5-sequential_1/lstm_2/while/Identity_5:output:0"?
=sequential_1_lstm_2_while_lstm_cell_2_readvariableop_resource?sequential_1_lstm_2_while_lstm_cell_2_readvariableop_resource_0"?
Esequential_1_lstm_2_while_lstm_cell_2_split_1_readvariableop_resourceGsequential_1_lstm_2_while_lstm_cell_2_split_1_readvariableop_resource_0"?
Csequential_1_lstm_2_while_lstm_cell_2_split_readvariableop_resourceEsequential_1_lstm_2_while_lstm_cell_2_split_readvariableop_resource_0"?
=sequential_1_lstm_2_while_sequential_1_lstm_2_strided_slice_1?sequential_1_lstm_2_while_sequential_1_lstm_2_strided_slice_1_0"?
ysequential_1_lstm_2_while_tensorarrayv2read_tensorlistgetitem_sequential_1_lstm_2_tensorarrayunstack_tensorlistfromtensor{sequential_1_lstm_2_while_tensorarrayv2read_tensorlistgetitem_sequential_1_lstm_2_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2l
4sequential_1/lstm_2/while/lstm_cell_2/ReadVariableOp4sequential_1/lstm_2/while/lstm_cell_2/ReadVariableOp2p
6sequential_1/lstm_2/while/lstm_cell_2/ReadVariableOp_16sequential_1/lstm_2/while/lstm_cell_2/ReadVariableOp_12p
6sequential_1/lstm_2/while/lstm_cell_2/ReadVariableOp_26sequential_1/lstm_2/while/lstm_cell_2/ReadVariableOp_22p
6sequential_1/lstm_2/while/lstm_cell_2/ReadVariableOp_36sequential_1/lstm_2/while/lstm_cell_2/ReadVariableOp_32x
:sequential_1/lstm_2/while/lstm_cell_2/split/ReadVariableOp:sequential_1/lstm_2/while/lstm_cell_2/split/ReadVariableOp2|
<sequential_1/lstm_2/while/lstm_cell_2/split_1/ReadVariableOp<sequential_1/lstm_2/while/lstm_cell_2/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?
?
while_cond_55252
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_55252___redundant_placeholder03
/while_while_cond_55252___redundant_placeholder13
/while_while_cond_55252___redundant_placeholder23
/while_while_cond_55252___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?
?
$sequential_1_lstm_3_while_cond_53789D
@sequential_1_lstm_3_while_sequential_1_lstm_3_while_loop_counterJ
Fsequential_1_lstm_3_while_sequential_1_lstm_3_while_maximum_iterations)
%sequential_1_lstm_3_while_placeholder+
'sequential_1_lstm_3_while_placeholder_1+
'sequential_1_lstm_3_while_placeholder_2+
'sequential_1_lstm_3_while_placeholder_3F
Bsequential_1_lstm_3_while_less_sequential_1_lstm_3_strided_slice_1[
Wsequential_1_lstm_3_while_sequential_1_lstm_3_while_cond_53789___redundant_placeholder0[
Wsequential_1_lstm_3_while_sequential_1_lstm_3_while_cond_53789___redundant_placeholder1[
Wsequential_1_lstm_3_while_sequential_1_lstm_3_while_cond_53789___redundant_placeholder2[
Wsequential_1_lstm_3_while_sequential_1_lstm_3_while_cond_53789___redundant_placeholder3&
"sequential_1_lstm_3_while_identity
?
sequential_1/lstm_3/while/LessLess%sequential_1_lstm_3_while_placeholderBsequential_1_lstm_3_while_less_sequential_1_lstm_3_strided_slice_1*
T0*
_output_shapes
: s
"sequential_1/lstm_3/while/IdentityIdentity"sequential_1/lstm_3/while/Less:z:0*
T0
*
_output_shapes
: "Q
"sequential_1_lstm_3_while_identity+sequential_1/lstm_3/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?
?
while_cond_55607
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_55607___redundant_placeholder03
/while_while_cond_55607___redundant_placeholder13
/while_while_cond_55607___redundant_placeholder23
/while_while_cond_55607___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
??
?
lstm_2_while_body_56343*
&lstm_2_while_lstm_2_while_loop_counter0
,lstm_2_while_lstm_2_while_maximum_iterations
lstm_2_while_placeholder
lstm_2_while_placeholder_1
lstm_2_while_placeholder_2
lstm_2_while_placeholder_3)
%lstm_2_while_lstm_2_strided_slice_1_0e
alstm_2_while_tensorarrayv2read_tensorlistgetitem_lstm_2_tensorarrayunstack_tensorlistfromtensor_0K
8lstm_2_while_lstm_cell_2_split_readvariableop_resource_0:	?I
:lstm_2_while_lstm_cell_2_split_1_readvariableop_resource_0:	?F
2lstm_2_while_lstm_cell_2_readvariableop_resource_0:
??
lstm_2_while_identity
lstm_2_while_identity_1
lstm_2_while_identity_2
lstm_2_while_identity_3
lstm_2_while_identity_4
lstm_2_while_identity_5'
#lstm_2_while_lstm_2_strided_slice_1c
_lstm_2_while_tensorarrayv2read_tensorlistgetitem_lstm_2_tensorarrayunstack_tensorlistfromtensorI
6lstm_2_while_lstm_cell_2_split_readvariableop_resource:	?G
8lstm_2_while_lstm_cell_2_split_1_readvariableop_resource:	?D
0lstm_2_while_lstm_cell_2_readvariableop_resource:
????'lstm_2/while/lstm_cell_2/ReadVariableOp?)lstm_2/while/lstm_cell_2/ReadVariableOp_1?)lstm_2/while/lstm_cell_2/ReadVariableOp_2?)lstm_2/while/lstm_cell_2/ReadVariableOp_3?-lstm_2/while/lstm_cell_2/split/ReadVariableOp?/lstm_2/while/lstm_cell_2/split_1/ReadVariableOp?
>lstm_2/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
0lstm_2/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemalstm_2_while_tensorarrayv2read_tensorlistgetitem_lstm_2_tensorarrayunstack_tensorlistfromtensor_0lstm_2_while_placeholderGlstm_2/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0j
(lstm_2/while/lstm_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
-lstm_2/while/lstm_cell_2/split/ReadVariableOpReadVariableOp8lstm_2_while_lstm_cell_2_split_readvariableop_resource_0*
_output_shapes
:	?*
dtype0?
lstm_2/while/lstm_cell_2/splitSplit1lstm_2/while/lstm_cell_2/split/split_dim:output:05lstm_2/while/lstm_cell_2/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	?:	?:	?:	?*
	num_split?
lstm_2/while/lstm_cell_2/MatMulMatMul7lstm_2/while/TensorArrayV2Read/TensorListGetItem:item:0'lstm_2/while/lstm_cell_2/split:output:0*
T0*(
_output_shapes
:???????????
!lstm_2/while/lstm_cell_2/MatMul_1MatMul7lstm_2/while/TensorArrayV2Read/TensorListGetItem:item:0'lstm_2/while/lstm_cell_2/split:output:1*
T0*(
_output_shapes
:???????????
!lstm_2/while/lstm_cell_2/MatMul_2MatMul7lstm_2/while/TensorArrayV2Read/TensorListGetItem:item:0'lstm_2/while/lstm_cell_2/split:output:2*
T0*(
_output_shapes
:???????????
!lstm_2/while/lstm_cell_2/MatMul_3MatMul7lstm_2/while/TensorArrayV2Read/TensorListGetItem:item:0'lstm_2/while/lstm_cell_2/split:output:3*
T0*(
_output_shapes
:??????????l
*lstm_2/while/lstm_cell_2/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
/lstm_2/while/lstm_cell_2/split_1/ReadVariableOpReadVariableOp:lstm_2_while_lstm_cell_2_split_1_readvariableop_resource_0*
_output_shapes	
:?*
dtype0?
 lstm_2/while/lstm_cell_2/split_1Split3lstm_2/while/lstm_cell_2/split_1/split_dim:output:07lstm_2/while/lstm_cell_2/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split?
 lstm_2/while/lstm_cell_2/BiasAddBiasAdd)lstm_2/while/lstm_cell_2/MatMul:product:0)lstm_2/while/lstm_cell_2/split_1:output:0*
T0*(
_output_shapes
:???????????
"lstm_2/while/lstm_cell_2/BiasAdd_1BiasAdd+lstm_2/while/lstm_cell_2/MatMul_1:product:0)lstm_2/while/lstm_cell_2/split_1:output:1*
T0*(
_output_shapes
:???????????
"lstm_2/while/lstm_cell_2/BiasAdd_2BiasAdd+lstm_2/while/lstm_cell_2/MatMul_2:product:0)lstm_2/while/lstm_cell_2/split_1:output:2*
T0*(
_output_shapes
:???????????
"lstm_2/while/lstm_cell_2/BiasAdd_3BiasAdd+lstm_2/while/lstm_cell_2/MatMul_3:product:0)lstm_2/while/lstm_cell_2/split_1:output:3*
T0*(
_output_shapes
:???????????
'lstm_2/while/lstm_cell_2/ReadVariableOpReadVariableOp2lstm_2_while_lstm_cell_2_readvariableop_resource_0* 
_output_shapes
:
??*
dtype0}
,lstm_2/while/lstm_cell_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        
.lstm_2/while/lstm_cell_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
.lstm_2/while/lstm_cell_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
&lstm_2/while/lstm_cell_2/strided_sliceStridedSlice/lstm_2/while/lstm_cell_2/ReadVariableOp:value:05lstm_2/while/lstm_cell_2/strided_slice/stack:output:07lstm_2/while/lstm_cell_2/strided_slice/stack_1:output:07lstm_2/while/lstm_cell_2/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
!lstm_2/while/lstm_cell_2/MatMul_4MatMullstm_2_while_placeholder_2/lstm_2/while/lstm_cell_2/strided_slice:output:0*
T0*(
_output_shapes
:???????????
lstm_2/while/lstm_cell_2/addAddV2)lstm_2/while/lstm_cell_2/BiasAdd:output:0+lstm_2/while/lstm_cell_2/MatMul_4:product:0*
T0*(
_output_shapes
:??????????c
lstm_2/while/lstm_cell_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>e
 lstm_2/while/lstm_cell_2/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ??
lstm_2/while/lstm_cell_2/MulMul lstm_2/while/lstm_cell_2/add:z:0'lstm_2/while/lstm_cell_2/Const:output:0*
T0*(
_output_shapes
:???????????
lstm_2/while/lstm_cell_2/Add_1AddV2 lstm_2/while/lstm_cell_2/Mul:z:0)lstm_2/while/lstm_cell_2/Const_1:output:0*
T0*(
_output_shapes
:??????????u
0lstm_2/while/lstm_cell_2/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
.lstm_2/while/lstm_cell_2/clip_by_value/MinimumMinimum"lstm_2/while/lstm_cell_2/Add_1:z:09lstm_2/while/lstm_cell_2/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:??????????m
(lstm_2/while/lstm_cell_2/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
&lstm_2/while/lstm_cell_2/clip_by_valueMaximum2lstm_2/while/lstm_cell_2/clip_by_value/Minimum:z:01lstm_2/while/lstm_cell_2/clip_by_value/y:output:0*
T0*(
_output_shapes
:???????????
)lstm_2/while/lstm_cell_2/ReadVariableOp_1ReadVariableOp2lstm_2_while_lstm_cell_2_readvariableop_resource_0* 
_output_shapes
:
??*
dtype0
.lstm_2/while/lstm_cell_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       ?
0lstm_2/while/lstm_cell_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
0lstm_2/while/lstm_cell_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
(lstm_2/while/lstm_cell_2/strided_slice_1StridedSlice1lstm_2/while/lstm_cell_2/ReadVariableOp_1:value:07lstm_2/while/lstm_cell_2/strided_slice_1/stack:output:09lstm_2/while/lstm_cell_2/strided_slice_1/stack_1:output:09lstm_2/while/lstm_cell_2/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
!lstm_2/while/lstm_cell_2/MatMul_5MatMullstm_2_while_placeholder_21lstm_2/while/lstm_cell_2/strided_slice_1:output:0*
T0*(
_output_shapes
:???????????
lstm_2/while/lstm_cell_2/add_2AddV2+lstm_2/while/lstm_cell_2/BiasAdd_1:output:0+lstm_2/while/lstm_cell_2/MatMul_5:product:0*
T0*(
_output_shapes
:??????????e
 lstm_2/while/lstm_cell_2/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>e
 lstm_2/while/lstm_cell_2/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ??
lstm_2/while/lstm_cell_2/Mul_1Mul"lstm_2/while/lstm_cell_2/add_2:z:0)lstm_2/while/lstm_cell_2/Const_2:output:0*
T0*(
_output_shapes
:???????????
lstm_2/while/lstm_cell_2/Add_3AddV2"lstm_2/while/lstm_cell_2/Mul_1:z:0)lstm_2/while/lstm_cell_2/Const_3:output:0*
T0*(
_output_shapes
:??????????w
2lstm_2/while/lstm_cell_2/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
0lstm_2/while/lstm_cell_2/clip_by_value_1/MinimumMinimum"lstm_2/while/lstm_cell_2/Add_3:z:0;lstm_2/while/lstm_cell_2/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:??????????o
*lstm_2/while/lstm_cell_2/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
(lstm_2/while/lstm_cell_2/clip_by_value_1Maximum4lstm_2/while/lstm_cell_2/clip_by_value_1/Minimum:z:03lstm_2/while/lstm_cell_2/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:???????????
lstm_2/while/lstm_cell_2/mul_2Mul,lstm_2/while/lstm_cell_2/clip_by_value_1:z:0lstm_2_while_placeholder_3*
T0*(
_output_shapes
:???????????
)lstm_2/while/lstm_cell_2/ReadVariableOp_2ReadVariableOp2lstm_2_while_lstm_cell_2_readvariableop_resource_0* 
_output_shapes
:
??*
dtype0
.lstm_2/while/lstm_cell_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       ?
0lstm_2/while/lstm_cell_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
0lstm_2/while/lstm_cell_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
(lstm_2/while/lstm_cell_2/strided_slice_2StridedSlice1lstm_2/while/lstm_cell_2/ReadVariableOp_2:value:07lstm_2/while/lstm_cell_2/strided_slice_2/stack:output:09lstm_2/while/lstm_cell_2/strided_slice_2/stack_1:output:09lstm_2/while/lstm_cell_2/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
!lstm_2/while/lstm_cell_2/MatMul_6MatMullstm_2_while_placeholder_21lstm_2/while/lstm_cell_2/strided_slice_2:output:0*
T0*(
_output_shapes
:???????????
lstm_2/while/lstm_cell_2/add_4AddV2+lstm_2/while/lstm_cell_2/BiasAdd_2:output:0+lstm_2/while/lstm_cell_2/MatMul_6:product:0*
T0*(
_output_shapes
:??????????|
lstm_2/while/lstm_cell_2/TanhTanh"lstm_2/while/lstm_cell_2/add_4:z:0*
T0*(
_output_shapes
:???????????
lstm_2/while/lstm_cell_2/mul_3Mul*lstm_2/while/lstm_cell_2/clip_by_value:z:0!lstm_2/while/lstm_cell_2/Tanh:y:0*
T0*(
_output_shapes
:???????????
lstm_2/while/lstm_cell_2/add_5AddV2"lstm_2/while/lstm_cell_2/mul_2:z:0"lstm_2/while/lstm_cell_2/mul_3:z:0*
T0*(
_output_shapes
:???????????
)lstm_2/while/lstm_cell_2/ReadVariableOp_3ReadVariableOp2lstm_2_while_lstm_cell_2_readvariableop_resource_0* 
_output_shapes
:
??*
dtype0
.lstm_2/while/lstm_cell_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       ?
0lstm_2/while/lstm_cell_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        ?
0lstm_2/while/lstm_cell_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
(lstm_2/while/lstm_cell_2/strided_slice_3StridedSlice1lstm_2/while/lstm_cell_2/ReadVariableOp_3:value:07lstm_2/while/lstm_cell_2/strided_slice_3/stack:output:09lstm_2/while/lstm_cell_2/strided_slice_3/stack_1:output:09lstm_2/while/lstm_cell_2/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
!lstm_2/while/lstm_cell_2/MatMul_7MatMullstm_2_while_placeholder_21lstm_2/while/lstm_cell_2/strided_slice_3:output:0*
T0*(
_output_shapes
:???????????
lstm_2/while/lstm_cell_2/add_6AddV2+lstm_2/while/lstm_cell_2/BiasAdd_3:output:0+lstm_2/while/lstm_cell_2/MatMul_7:product:0*
T0*(
_output_shapes
:??????????e
 lstm_2/while/lstm_cell_2/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>e
 lstm_2/while/lstm_cell_2/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ??
lstm_2/while/lstm_cell_2/Mul_4Mul"lstm_2/while/lstm_cell_2/add_6:z:0)lstm_2/while/lstm_cell_2/Const_4:output:0*
T0*(
_output_shapes
:???????????
lstm_2/while/lstm_cell_2/Add_7AddV2"lstm_2/while/lstm_cell_2/Mul_4:z:0)lstm_2/while/lstm_cell_2/Const_5:output:0*
T0*(
_output_shapes
:??????????w
2lstm_2/while/lstm_cell_2/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
0lstm_2/while/lstm_cell_2/clip_by_value_2/MinimumMinimum"lstm_2/while/lstm_cell_2/Add_7:z:0;lstm_2/while/lstm_cell_2/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:??????????o
*lstm_2/while/lstm_cell_2/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
(lstm_2/while/lstm_cell_2/clip_by_value_2Maximum4lstm_2/while/lstm_cell_2/clip_by_value_2/Minimum:z:03lstm_2/while/lstm_cell_2/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:??????????~
lstm_2/while/lstm_cell_2/Tanh_1Tanh"lstm_2/while/lstm_cell_2/add_5:z:0*
T0*(
_output_shapes
:???????????
lstm_2/while/lstm_cell_2/mul_5Mul,lstm_2/while/lstm_cell_2/clip_by_value_2:z:0#lstm_2/while/lstm_cell_2/Tanh_1:y:0*
T0*(
_output_shapes
:???????????
1lstm_2/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_2_while_placeholder_1lstm_2_while_placeholder"lstm_2/while/lstm_cell_2/mul_5:z:0*
_output_shapes
: *
element_dtype0:???T
lstm_2/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :q
lstm_2/while/addAddV2lstm_2_while_placeholderlstm_2/while/add/y:output:0*
T0*
_output_shapes
: V
lstm_2/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :?
lstm_2/while/add_1AddV2&lstm_2_while_lstm_2_while_loop_counterlstm_2/while/add_1/y:output:0*
T0*
_output_shapes
: n
lstm_2/while/IdentityIdentitylstm_2/while/add_1:z:0^lstm_2/while/NoOp*
T0*
_output_shapes
: ?
lstm_2/while/Identity_1Identity,lstm_2_while_lstm_2_while_maximum_iterations^lstm_2/while/NoOp*
T0*
_output_shapes
: n
lstm_2/while/Identity_2Identitylstm_2/while/add:z:0^lstm_2/while/NoOp*
T0*
_output_shapes
: ?
lstm_2/while/Identity_3IdentityAlstm_2/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_2/while/NoOp*
T0*
_output_shapes
: ?
lstm_2/while/Identity_4Identity"lstm_2/while/lstm_cell_2/mul_5:z:0^lstm_2/while/NoOp*
T0*(
_output_shapes
:???????????
lstm_2/while/Identity_5Identity"lstm_2/while/lstm_cell_2/add_5:z:0^lstm_2/while/NoOp*
T0*(
_output_shapes
:???????????
lstm_2/while/NoOpNoOp(^lstm_2/while/lstm_cell_2/ReadVariableOp*^lstm_2/while/lstm_cell_2/ReadVariableOp_1*^lstm_2/while/lstm_cell_2/ReadVariableOp_2*^lstm_2/while/lstm_cell_2/ReadVariableOp_3.^lstm_2/while/lstm_cell_2/split/ReadVariableOp0^lstm_2/while/lstm_cell_2/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "7
lstm_2_while_identitylstm_2/while/Identity:output:0";
lstm_2_while_identity_1 lstm_2/while/Identity_1:output:0";
lstm_2_while_identity_2 lstm_2/while/Identity_2:output:0";
lstm_2_while_identity_3 lstm_2/while/Identity_3:output:0";
lstm_2_while_identity_4 lstm_2/while/Identity_4:output:0";
lstm_2_while_identity_5 lstm_2/while/Identity_5:output:0"L
#lstm_2_while_lstm_2_strided_slice_1%lstm_2_while_lstm_2_strided_slice_1_0"f
0lstm_2_while_lstm_cell_2_readvariableop_resource2lstm_2_while_lstm_cell_2_readvariableop_resource_0"v
8lstm_2_while_lstm_cell_2_split_1_readvariableop_resource:lstm_2_while_lstm_cell_2_split_1_readvariableop_resource_0"r
6lstm_2_while_lstm_cell_2_split_readvariableop_resource8lstm_2_while_lstm_cell_2_split_readvariableop_resource_0"?
_lstm_2_while_tensorarrayv2read_tensorlistgetitem_lstm_2_tensorarrayunstack_tensorlistfromtensoralstm_2_while_tensorarrayv2read_tensorlistgetitem_lstm_2_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2R
'lstm_2/while/lstm_cell_2/ReadVariableOp'lstm_2/while/lstm_cell_2/ReadVariableOp2V
)lstm_2/while/lstm_cell_2/ReadVariableOp_1)lstm_2/while/lstm_cell_2/ReadVariableOp_12V
)lstm_2/while/lstm_cell_2/ReadVariableOp_2)lstm_2/while/lstm_cell_2/ReadVariableOp_22V
)lstm_2/while/lstm_cell_2/ReadVariableOp_3)lstm_2/while/lstm_cell_2/ReadVariableOp_32^
-lstm_2/while/lstm_cell_2/split/ReadVariableOp-lstm_2/while/lstm_cell_2/split/ReadVariableOp2b
/lstm_2/while/lstm_cell_2/split_1/ReadVariableOp/lstm_2/while/lstm_cell_2/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?J
?
F__inference_lstm_cell_2_layer_call_and_return_conditional_losses_54068

inputs

states
states_10
split_readvariableop_resource:	?.
split_1_readvariableop_resource:	?+
readvariableop_resource:
??
identity

identity_1

identity_2??ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?ReadVariableOp_3?split/ReadVariableOp?split_1/ReadVariableOpQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :s
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes
:	?*
dtype0?
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	?:	?:	?:	?*
	num_split[
MatMulMatMulinputssplit:output:0*
T0*(
_output_shapes
:??????????]
MatMul_1MatMulinputssplit:output:1*
T0*(
_output_shapes
:??????????]
MatMul_2MatMulinputssplit:output:2*
T0*(
_output_shapes
:??????????]
MatMul_3MatMulinputssplit:output:3*
T0*(
_output_shapes
:??????????S
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : s
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes	
:?*
dtype0?
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_spliti
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*(
_output_shapes
:??????????m
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*(
_output_shapes
:??????????m
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*(
_output_shapes
:??????????m
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*(
_output_shapes
:??????????h
ReadVariableOpReadVariableOpreadvariableop_resource* 
_output_shapes
:
??*
dtype0d
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_maske
MatMul_4MatMulstatesstrided_slice:output:0*
T0*(
_output_shapes
:??????????e
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*(
_output_shapes
:??????????J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>L
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?V
MulMuladd:z:0Const:output:0*
T0*(
_output_shapes
:??????????\
Add_1AddV2Mul:z:0Const_1:output:0*
T0*(
_output_shapes
:??????????\
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:??????????T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*(
_output_shapes
:??????????j
ReadVariableOp_1ReadVariableOpreadvariableop_resource* 
_output_shapes
:
??*
dtype0f
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_maskg
MatMul_5MatMulstatesstrided_slice_1:output:0*
T0*(
_output_shapes
:??????????i
add_2AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*(
_output_shapes
:??????????L
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>L
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?\
Mul_1Mul	add_2:z:0Const_2:output:0*
T0*(
_output_shapes
:??????????^
Add_3AddV2	Mul_1:z:0Const_3:output:0*
T0*(
_output_shapes
:??????????^
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:??????????V
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*(
_output_shapes
:??????????^
mul_2Mulclip_by_value_1:z:0states_1*
T0*(
_output_shapes
:??????????j
ReadVariableOp_2ReadVariableOpreadvariableop_resource* 
_output_shapes
:
??*
dtype0f
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_maskg
MatMul_6MatMulstatesstrided_slice_2:output:0*
T0*(
_output_shapes
:??????????i
add_4AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*(
_output_shapes
:??????????J
TanhTanh	add_4:z:0*
T0*(
_output_shapes
:??????????\
mul_3Mulclip_by_value:z:0Tanh:y:0*
T0*(
_output_shapes
:??????????W
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*(
_output_shapes
:??????????j
ReadVariableOp_3ReadVariableOpreadvariableop_resource* 
_output_shapes
:
??*
dtype0f
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_maskg
MatMul_7MatMulstatesstrided_slice_3:output:0*
T0*(
_output_shapes
:??????????i
add_6AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*(
_output_shapes
:??????????L
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>L
Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?\
Mul_4Mul	add_6:z:0Const_4:output:0*
T0*(
_output_shapes
:??????????^
Add_7AddV2	Mul_4:z:0Const_5:output:0*
T0*(
_output_shapes
:??????????^
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:??????????V
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*(
_output_shapes
:??????????L
Tanh_1Tanh	add_5:z:0*
T0*(
_output_shapes
:??????????`
mul_5Mulclip_by_value_2:z:0
Tanh_1:y:0*
T0*(
_output_shapes
:??????????Y
IdentityIdentity	mul_5:z:0^NoOp*
T0*(
_output_shapes
:??????????[

Identity_1Identity	mul_5:z:0^NoOp*
T0*(
_output_shapes
:??????????[

Identity_2Identity	add_5:z:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:?????????:??????????:??????????: : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:PL
(
_output_shapes
:??????????
 
_user_specified_namestates:PL
(
_output_shapes
:??????????
 
_user_specified_namestates
??
?
A__inference_lstm_2_layer_call_and_return_conditional_losses_57577
inputs_0<
)lstm_cell_2_split_readvariableop_resource:	?:
+lstm_cell_2_split_1_readvariableop_resource:	?7
#lstm_cell_2_readvariableop_resource:
??
identity??lstm_cell_2/ReadVariableOp?lstm_cell_2/ReadVariableOp_1?lstm_cell_2/ReadVariableOp_2?lstm_cell_2/ReadVariableOp_3? lstm_cell_2/split/ReadVariableOp?"lstm_cell_2/split_1/ReadVariableOp?while=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????S
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask]
lstm_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
 lstm_cell_2/split/ReadVariableOpReadVariableOp)lstm_cell_2_split_readvariableop_resource*
_output_shapes
:	?*
dtype0?
lstm_cell_2/splitSplit$lstm_cell_2/split/split_dim:output:0(lstm_cell_2/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	?:	?:	?:	?*
	num_split?
lstm_cell_2/MatMulMatMulstrided_slice_2:output:0lstm_cell_2/split:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_2/MatMul_1MatMulstrided_slice_2:output:0lstm_cell_2/split:output:1*
T0*(
_output_shapes
:???????????
lstm_cell_2/MatMul_2MatMulstrided_slice_2:output:0lstm_cell_2/split:output:2*
T0*(
_output_shapes
:???????????
lstm_cell_2/MatMul_3MatMulstrided_slice_2:output:0lstm_cell_2/split:output:3*
T0*(
_output_shapes
:??????????_
lstm_cell_2/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
"lstm_cell_2/split_1/ReadVariableOpReadVariableOp+lstm_cell_2_split_1_readvariableop_resource*
_output_shapes	
:?*
dtype0?
lstm_cell_2/split_1Split&lstm_cell_2/split_1/split_dim:output:0*lstm_cell_2/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split?
lstm_cell_2/BiasAddBiasAddlstm_cell_2/MatMul:product:0lstm_cell_2/split_1:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_2/BiasAdd_1BiasAddlstm_cell_2/MatMul_1:product:0lstm_cell_2/split_1:output:1*
T0*(
_output_shapes
:???????????
lstm_cell_2/BiasAdd_2BiasAddlstm_cell_2/MatMul_2:product:0lstm_cell_2/split_1:output:2*
T0*(
_output_shapes
:???????????
lstm_cell_2/BiasAdd_3BiasAddlstm_cell_2/MatMul_3:product:0lstm_cell_2/split_1:output:3*
T0*(
_output_shapes
:???????????
lstm_cell_2/ReadVariableOpReadVariableOp#lstm_cell_2_readvariableop_resource* 
_output_shapes
:
??*
dtype0p
lstm_cell_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        r
!lstm_cell_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       r
!lstm_cell_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
lstm_cell_2/strided_sliceStridedSlice"lstm_cell_2/ReadVariableOp:value:0(lstm_cell_2/strided_slice/stack:output:0*lstm_cell_2/strided_slice/stack_1:output:0*lstm_cell_2/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
lstm_cell_2/MatMul_4MatMulzeros:output:0"lstm_cell_2/strided_slice:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_2/addAddV2lstm_cell_2/BiasAdd:output:0lstm_cell_2/MatMul_4:product:0*
T0*(
_output_shapes
:??????????V
lstm_cell_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>X
lstm_cell_2/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?z
lstm_cell_2/MulMullstm_cell_2/add:z:0lstm_cell_2/Const:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_2/Add_1AddV2lstm_cell_2/Mul:z:0lstm_cell_2/Const_1:output:0*
T0*(
_output_shapes
:??????????h
#lstm_cell_2/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
!lstm_cell_2/clip_by_value/MinimumMinimumlstm_cell_2/Add_1:z:0,lstm_cell_2/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:??????????`
lstm_cell_2/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
lstm_cell_2/clip_by_valueMaximum%lstm_cell_2/clip_by_value/Minimum:z:0$lstm_cell_2/clip_by_value/y:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_2/ReadVariableOp_1ReadVariableOp#lstm_cell_2_readvariableop_resource* 
_output_shapes
:
??*
dtype0r
!lstm_cell_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       t
#lstm_cell_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       t
#lstm_cell_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
lstm_cell_2/strided_slice_1StridedSlice$lstm_cell_2/ReadVariableOp_1:value:0*lstm_cell_2/strided_slice_1/stack:output:0,lstm_cell_2/strided_slice_1/stack_1:output:0,lstm_cell_2/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
lstm_cell_2/MatMul_5MatMulzeros:output:0$lstm_cell_2/strided_slice_1:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_2/add_2AddV2lstm_cell_2/BiasAdd_1:output:0lstm_cell_2/MatMul_5:product:0*
T0*(
_output_shapes
:??????????X
lstm_cell_2/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>X
lstm_cell_2/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ??
lstm_cell_2/Mul_1Mullstm_cell_2/add_2:z:0lstm_cell_2/Const_2:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_2/Add_3AddV2lstm_cell_2/Mul_1:z:0lstm_cell_2/Const_3:output:0*
T0*(
_output_shapes
:??????????j
%lstm_cell_2/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
#lstm_cell_2/clip_by_value_1/MinimumMinimumlstm_cell_2/Add_3:z:0.lstm_cell_2/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:??????????b
lstm_cell_2/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
lstm_cell_2/clip_by_value_1Maximum'lstm_cell_2/clip_by_value_1/Minimum:z:0&lstm_cell_2/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:??????????~
lstm_cell_2/mul_2Mullstm_cell_2/clip_by_value_1:z:0zeros_1:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_2/ReadVariableOp_2ReadVariableOp#lstm_cell_2_readvariableop_resource* 
_output_shapes
:
??*
dtype0r
!lstm_cell_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       t
#lstm_cell_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       t
#lstm_cell_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
lstm_cell_2/strided_slice_2StridedSlice$lstm_cell_2/ReadVariableOp_2:value:0*lstm_cell_2/strided_slice_2/stack:output:0,lstm_cell_2/strided_slice_2/stack_1:output:0,lstm_cell_2/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
lstm_cell_2/MatMul_6MatMulzeros:output:0$lstm_cell_2/strided_slice_2:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_2/add_4AddV2lstm_cell_2/BiasAdd_2:output:0lstm_cell_2/MatMul_6:product:0*
T0*(
_output_shapes
:??????????b
lstm_cell_2/TanhTanhlstm_cell_2/add_4:z:0*
T0*(
_output_shapes
:???????????
lstm_cell_2/mul_3Mullstm_cell_2/clip_by_value:z:0lstm_cell_2/Tanh:y:0*
T0*(
_output_shapes
:??????????{
lstm_cell_2/add_5AddV2lstm_cell_2/mul_2:z:0lstm_cell_2/mul_3:z:0*
T0*(
_output_shapes
:???????????
lstm_cell_2/ReadVariableOp_3ReadVariableOp#lstm_cell_2_readvariableop_resource* 
_output_shapes
:
??*
dtype0r
!lstm_cell_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       t
#lstm_cell_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        t
#lstm_cell_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
lstm_cell_2/strided_slice_3StridedSlice$lstm_cell_2/ReadVariableOp_3:value:0*lstm_cell_2/strided_slice_3/stack:output:0,lstm_cell_2/strided_slice_3/stack_1:output:0,lstm_cell_2/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
lstm_cell_2/MatMul_7MatMulzeros:output:0$lstm_cell_2/strided_slice_3:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_2/add_6AddV2lstm_cell_2/BiasAdd_3:output:0lstm_cell_2/MatMul_7:product:0*
T0*(
_output_shapes
:??????????X
lstm_cell_2/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>X
lstm_cell_2/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ??
lstm_cell_2/Mul_4Mullstm_cell_2/add_6:z:0lstm_cell_2/Const_4:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_2/Add_7AddV2lstm_cell_2/Mul_4:z:0lstm_cell_2/Const_5:output:0*
T0*(
_output_shapes
:??????????j
%lstm_cell_2/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
#lstm_cell_2/clip_by_value_2/MinimumMinimumlstm_cell_2/Add_7:z:0.lstm_cell_2/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:??????????b
lstm_cell_2/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
lstm_cell_2/clip_by_value_2Maximum'lstm_cell_2/clip_by_value_2/Minimum:z:0&lstm_cell_2/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:??????????d
lstm_cell_2/Tanh_1Tanhlstm_cell_2/add_5:z:0*
T0*(
_output_shapes
:???????????
lstm_cell_2/mul_5Mullstm_cell_2/clip_by_value_2:z:0lstm_cell_2/Tanh_1:y:0*
T0*(
_output_shapes
:??????????n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)lstm_cell_2_split_readvariableop_resource+lstm_cell_2_split_1_readvariableop_resource#lstm_cell_2_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_57437*
condR
while_cond_57436*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:???????????????????*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:???????????????????l
IdentityIdentitytranspose_1:y:0^NoOp*
T0*5
_output_shapes#
!:????????????????????
NoOpNoOp^lstm_cell_2/ReadVariableOp^lstm_cell_2/ReadVariableOp_1^lstm_cell_2/ReadVariableOp_2^lstm_cell_2/ReadVariableOp_3!^lstm_cell_2/split/ReadVariableOp#^lstm_cell_2/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????: : : 28
lstm_cell_2/ReadVariableOplstm_cell_2/ReadVariableOp2<
lstm_cell_2/ReadVariableOp_1lstm_cell_2/ReadVariableOp_12<
lstm_cell_2/ReadVariableOp_2lstm_cell_2/ReadVariableOp_22<
lstm_cell_2/ReadVariableOp_3lstm_cell_2/ReadVariableOp_32D
 lstm_cell_2/split/ReadVariableOp lstm_cell_2/split/ReadVariableOp2H
"lstm_cell_2/split_1/ReadVariableOp"lstm_cell_2/split_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :??????????????????
"
_user_specified_name
inputs/0
??
?
A__inference_lstm_3_layer_call_and_return_conditional_losses_58901
inputs_0=
)lstm_cell_3_split_readvariableop_resource:
??:
+lstm_cell_3_split_1_readvariableop_resource:	?7
#lstm_cell_3_readvariableop_resource:
??
identity??lstm_cell_3/ReadVariableOp?lstm_cell_3/ReadVariableOp_1?lstm_cell_3/ReadVariableOp_2?lstm_cell_3/ReadVariableOp_3? lstm_cell_3/split/ReadVariableOp?"lstm_cell_3/split_1/ReadVariableOp?while=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????S
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          y
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*5
_output_shapes#
!:???????????????????D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask]
lstm_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
 lstm_cell_3/split/ReadVariableOpReadVariableOp)lstm_cell_3_split_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
lstm_cell_3/splitSplit$lstm_cell_3/split/split_dim:output:0(lstm_cell_3/split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
??:
??:
??:
??*
	num_split?
lstm_cell_3/MatMulMatMulstrided_slice_2:output:0lstm_cell_3/split:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_3/MatMul_1MatMulstrided_slice_2:output:0lstm_cell_3/split:output:1*
T0*(
_output_shapes
:???????????
lstm_cell_3/MatMul_2MatMulstrided_slice_2:output:0lstm_cell_3/split:output:2*
T0*(
_output_shapes
:???????????
lstm_cell_3/MatMul_3MatMulstrided_slice_2:output:0lstm_cell_3/split:output:3*
T0*(
_output_shapes
:??????????_
lstm_cell_3/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
"lstm_cell_3/split_1/ReadVariableOpReadVariableOp+lstm_cell_3_split_1_readvariableop_resource*
_output_shapes	
:?*
dtype0?
lstm_cell_3/split_1Split&lstm_cell_3/split_1/split_dim:output:0*lstm_cell_3/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split?
lstm_cell_3/BiasAddBiasAddlstm_cell_3/MatMul:product:0lstm_cell_3/split_1:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_3/BiasAdd_1BiasAddlstm_cell_3/MatMul_1:product:0lstm_cell_3/split_1:output:1*
T0*(
_output_shapes
:???????????
lstm_cell_3/BiasAdd_2BiasAddlstm_cell_3/MatMul_2:product:0lstm_cell_3/split_1:output:2*
T0*(
_output_shapes
:???????????
lstm_cell_3/BiasAdd_3BiasAddlstm_cell_3/MatMul_3:product:0lstm_cell_3/split_1:output:3*
T0*(
_output_shapes
:???????????
lstm_cell_3/ReadVariableOpReadVariableOp#lstm_cell_3_readvariableop_resource* 
_output_shapes
:
??*
dtype0p
lstm_cell_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        r
!lstm_cell_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   r
!lstm_cell_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
lstm_cell_3/strided_sliceStridedSlice"lstm_cell_3/ReadVariableOp:value:0(lstm_cell_3/strided_slice/stack:output:0*lstm_cell_3/strided_slice/stack_1:output:0*lstm_cell_3/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
lstm_cell_3/MatMul_4MatMulzeros:output:0"lstm_cell_3/strided_slice:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_3/addAddV2lstm_cell_3/BiasAdd:output:0lstm_cell_3/MatMul_4:product:0*
T0*(
_output_shapes
:??????????V
lstm_cell_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>X
lstm_cell_3/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?z
lstm_cell_3/MulMullstm_cell_3/add:z:0lstm_cell_3/Const:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_3/Add_1AddV2lstm_cell_3/Mul:z:0lstm_cell_3/Const_1:output:0*
T0*(
_output_shapes
:??????????h
#lstm_cell_3/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
!lstm_cell_3/clip_by_value/MinimumMinimumlstm_cell_3/Add_1:z:0,lstm_cell_3/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:??????????`
lstm_cell_3/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
lstm_cell_3/clip_by_valueMaximum%lstm_cell_3/clip_by_value/Minimum:z:0$lstm_cell_3/clip_by_value/y:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_3/ReadVariableOp_1ReadVariableOp#lstm_cell_3_readvariableop_resource* 
_output_shapes
:
??*
dtype0r
!lstm_cell_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   t
#lstm_cell_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       t
#lstm_cell_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
lstm_cell_3/strided_slice_1StridedSlice$lstm_cell_3/ReadVariableOp_1:value:0*lstm_cell_3/strided_slice_1/stack:output:0,lstm_cell_3/strided_slice_1/stack_1:output:0,lstm_cell_3/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
lstm_cell_3/MatMul_5MatMulzeros:output:0$lstm_cell_3/strided_slice_1:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_3/add_2AddV2lstm_cell_3/BiasAdd_1:output:0lstm_cell_3/MatMul_5:product:0*
T0*(
_output_shapes
:??????????X
lstm_cell_3/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>X
lstm_cell_3/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ??
lstm_cell_3/Mul_1Mullstm_cell_3/add_2:z:0lstm_cell_3/Const_2:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_3/Add_3AddV2lstm_cell_3/Mul_1:z:0lstm_cell_3/Const_3:output:0*
T0*(
_output_shapes
:??????????j
%lstm_cell_3/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
#lstm_cell_3/clip_by_value_1/MinimumMinimumlstm_cell_3/Add_3:z:0.lstm_cell_3/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:??????????b
lstm_cell_3/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
lstm_cell_3/clip_by_value_1Maximum'lstm_cell_3/clip_by_value_1/Minimum:z:0&lstm_cell_3/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:??????????~
lstm_cell_3/mul_2Mullstm_cell_3/clip_by_value_1:z:0zeros_1:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_3/ReadVariableOp_2ReadVariableOp#lstm_cell_3_readvariableop_resource* 
_output_shapes
:
??*
dtype0r
!lstm_cell_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       t
#lstm_cell_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  t
#lstm_cell_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
lstm_cell_3/strided_slice_2StridedSlice$lstm_cell_3/ReadVariableOp_2:value:0*lstm_cell_3/strided_slice_2/stack:output:0,lstm_cell_3/strided_slice_2/stack_1:output:0,lstm_cell_3/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
lstm_cell_3/MatMul_6MatMulzeros:output:0$lstm_cell_3/strided_slice_2:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_3/add_4AddV2lstm_cell_3/BiasAdd_2:output:0lstm_cell_3/MatMul_6:product:0*
T0*(
_output_shapes
:??????????b
lstm_cell_3/TanhTanhlstm_cell_3/add_4:z:0*
T0*(
_output_shapes
:???????????
lstm_cell_3/mul_3Mullstm_cell_3/clip_by_value:z:0lstm_cell_3/Tanh:y:0*
T0*(
_output_shapes
:??????????{
lstm_cell_3/add_5AddV2lstm_cell_3/mul_2:z:0lstm_cell_3/mul_3:z:0*
T0*(
_output_shapes
:???????????
lstm_cell_3/ReadVariableOp_3ReadVariableOp#lstm_cell_3_readvariableop_resource* 
_output_shapes
:
??*
dtype0r
!lstm_cell_3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  t
#lstm_cell_3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        t
#lstm_cell_3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
lstm_cell_3/strided_slice_3StridedSlice$lstm_cell_3/ReadVariableOp_3:value:0*lstm_cell_3/strided_slice_3/stack:output:0,lstm_cell_3/strided_slice_3/stack_1:output:0,lstm_cell_3/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
lstm_cell_3/MatMul_7MatMulzeros:output:0$lstm_cell_3/strided_slice_3:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_3/add_6AddV2lstm_cell_3/BiasAdd_3:output:0lstm_cell_3/MatMul_7:product:0*
T0*(
_output_shapes
:??????????X
lstm_cell_3/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>X
lstm_cell_3/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ??
lstm_cell_3/Mul_4Mullstm_cell_3/add_6:z:0lstm_cell_3/Const_4:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_3/Add_7AddV2lstm_cell_3/Mul_4:z:0lstm_cell_3/Const_5:output:0*
T0*(
_output_shapes
:??????????j
%lstm_cell_3/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
#lstm_cell_3/clip_by_value_2/MinimumMinimumlstm_cell_3/Add_7:z:0.lstm_cell_3/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:??????????b
lstm_cell_3/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
lstm_cell_3/clip_by_value_2Maximum'lstm_cell_3/clip_by_value_2/Minimum:z:0&lstm_cell_3/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:??????????d
lstm_cell_3/Tanh_1Tanhlstm_cell_3/add_5:z:0*
T0*(
_output_shapes
:???????????
lstm_cell_3/mul_5Mullstm_cell_3/clip_by_value_2:z:0lstm_cell_3/Tanh_1:y:0*
T0*(
_output_shapes
:??????????n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   ?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)lstm_cell_3_split_readvariableop_resource+lstm_cell_3_split_1_readvariableop_resource#lstm_cell_3_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_58761*
condR
while_cond_58760*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:???????????????????*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:???????????????????h
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp^lstm_cell_3/ReadVariableOp^lstm_cell_3/ReadVariableOp_1^lstm_cell_3/ReadVariableOp_2^lstm_cell_3/ReadVariableOp_3!^lstm_cell_3/split/ReadVariableOp#^lstm_cell_3/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':???????????????????: : : 28
lstm_cell_3/ReadVariableOplstm_cell_3/ReadVariableOp2<
lstm_cell_3/ReadVariableOp_1lstm_cell_3/ReadVariableOp_12<
lstm_cell_3/ReadVariableOp_2lstm_cell_3/ReadVariableOp_22<
lstm_cell_3/ReadVariableOp_3lstm_cell_3/ReadVariableOp_32D
 lstm_cell_3/split/ReadVariableOp lstm_cell_3/split/ReadVariableOp2H
"lstm_cell_3/split_1/ReadVariableOp"lstm_cell_3/split_1/ReadVariableOp2
whilewhile:_ [
5
_output_shapes#
!:???????????????????
"
_user_specified_name
inputs/0
?"
?
while_body_54082
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0,
while_lstm_cell_2_54106_0:	?(
while_lstm_cell_2_54108_0:	?-
while_lstm_cell_2_54110_0:
??
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor*
while_lstm_cell_2_54106:	?&
while_lstm_cell_2_54108:	?+
while_lstm_cell_2_54110:
????)while/lstm_cell_2/StatefulPartitionedCall?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0?
)while/lstm_cell_2/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_2_54106_0while_lstm_cell_2_54108_0while_lstm_cell_2_54110_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:??????????:??????????:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_lstm_cell_2_layer_call_and_return_conditional_losses_54068?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder2while/lstm_cell_2/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:???M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_4Identity2while/lstm_cell_2/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:???????????
while/Identity_5Identity2while/lstm_cell_2/StatefulPartitionedCall:output:2^while/NoOp*
T0*(
_output_shapes
:??????????x

while/NoOpNoOp*^while/lstm_cell_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"4
while_lstm_cell_2_54106while_lstm_cell_2_54106_0"4
while_lstm_cell_2_54108while_lstm_cell_2_54108_0"4
while_lstm_cell_2_54110while_lstm_cell_2_54110_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2V
)while/lstm_cell_2/StatefulPartitionedCall)while/lstm_cell_2/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?
?
while_cond_58504
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_58504___redundant_placeholder03
/while_while_cond_58504___redundant_placeholder13
/while_while_cond_58504___redundant_placeholder23
/while_while_cond_58504___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?
?
&__inference_lstm_2_layer_call_fn_58356
inputs_0
unknown:	?
	unknown_0:	?
	unknown_1:
??
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:???????????????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_lstm_2_layer_call_and_return_conditional_losses_54150}
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:???????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :??????????????????
"
_user_specified_name
inputs/0
?

?
,__inference_sequential_1_layer_call_fn_56138
lstm_2_input
unknown:	?
	unknown_0:	?
	unknown_1:
??
	unknown_2:
??
	unknown_3:	?
	unknown_4:
??
	unknown_5:
??
	unknown_6:	?
	unknown_7:	?
	unknown_8:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalllstm_2_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_56090o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
+
_output_shapes
:?????????
&
_user_specified_namelstm_2_input
?
?
while_cond_54328
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_54328___redundant_placeholder03
/while_while_cond_54328___redundant_placeholder13
/while_while_cond_54328___redundant_placeholder23
/while_while_cond_54328___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?
?
while_cond_54543
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_54543___redundant_placeholder03
/while_while_cond_54543___redundant_placeholder13
/while_while_cond_54543___redundant_placeholder23
/while_while_cond_54543___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?
?
'__inference_dense_2_layer_call_fn_59477

inputs
unknown:
??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_55412p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
&__inference_lstm_3_layer_call_fn_59424
inputs_0
unknown:
??
	unknown_0:	?
	unknown_1:
??
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_lstm_3_layer_call_and_return_conditional_losses_54612p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':???????????????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
5
_output_shapes#
!:???????????????????
"
_user_specified_name
inputs/0
?
?
while_cond_57948
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_57948___redundant_placeholder03
/while_while_cond_57948___redundant_placeholder13
/while_while_cond_57948___redundant_placeholder23
/while_while_cond_57948___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
??
?
!__inference__traced_restore_60188
file_prefix3
assignvariableop_dense_2_kernel:
??.
assignvariableop_1_dense_2_bias:	?4
!assignvariableop_2_dense_3_kernel:	?-
assignvariableop_3_dense_3_bias:&
assignvariableop_4_adam_iter:	 (
assignvariableop_5_adam_beta_1: (
assignvariableop_6_adam_beta_2: '
assignvariableop_7_adam_decay: /
%assignvariableop_8_adam_learning_rate: ?
,assignvariableop_9_lstm_2_lstm_cell_2_kernel:	?K
7assignvariableop_10_lstm_2_lstm_cell_2_recurrent_kernel:
??:
+assignvariableop_11_lstm_2_lstm_cell_2_bias:	?A
-assignvariableop_12_lstm_3_lstm_cell_3_kernel:
??K
7assignvariableop_13_lstm_3_lstm_cell_3_recurrent_kernel:
??:
+assignvariableop_14_lstm_3_lstm_cell_3_bias:	?#
assignvariableop_15_total: #
assignvariableop_16_count: %
assignvariableop_17_total_1: %
assignvariableop_18_count_1: =
)assignvariableop_19_adam_dense_2_kernel_m:
??6
'assignvariableop_20_adam_dense_2_bias_m:	?<
)assignvariableop_21_adam_dense_3_kernel_m:	?5
'assignvariableop_22_adam_dense_3_bias_m:G
4assignvariableop_23_adam_lstm_2_lstm_cell_2_kernel_m:	?R
>assignvariableop_24_adam_lstm_2_lstm_cell_2_recurrent_kernel_m:
??A
2assignvariableop_25_adam_lstm_2_lstm_cell_2_bias_m:	?H
4assignvariableop_26_adam_lstm_3_lstm_cell_3_kernel_m:
??R
>assignvariableop_27_adam_lstm_3_lstm_cell_3_recurrent_kernel_m:
??A
2assignvariableop_28_adam_lstm_3_lstm_cell_3_bias_m:	?=
)assignvariableop_29_adam_dense_2_kernel_v:
??6
'assignvariableop_30_adam_dense_2_bias_v:	?<
)assignvariableop_31_adam_dense_3_kernel_v:	?5
'assignvariableop_32_adam_dense_3_bias_v:G
4assignvariableop_33_adam_lstm_2_lstm_cell_2_kernel_v:	?R
>assignvariableop_34_adam_lstm_2_lstm_cell_2_recurrent_kernel_v:
??A
2assignvariableop_35_adam_lstm_2_lstm_cell_2_bias_v:	?H
4assignvariableop_36_adam_lstm_3_lstm_cell_3_kernel_v:
??R
>assignvariableop_37_adam_lstm_3_lstm_cell_3_recurrent_kernel_v:
??A
2assignvariableop_38_adam_lstm_3_lstm_cell_3_bias_v:	?
identity_40??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:(*
dtype0*?
value?B?(B6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:(*
dtype0*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::*6
dtypes,
*2(	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOpAssignVariableOpassignvariableop_dense_2_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_2_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_2AssignVariableOp!assignvariableop_2_dense_3_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOpassignvariableop_3_dense_3_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_iterIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_beta_1Identity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_beta_2Identity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_decayIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_8AssignVariableOp%assignvariableop_8_adam_learning_rateIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOp,assignvariableop_9_lstm_2_lstm_cell_2_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_10AssignVariableOp7assignvariableop_10_lstm_2_lstm_cell_2_recurrent_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOp+assignvariableop_11_lstm_2_lstm_cell_2_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_12AssignVariableOp-assignvariableop_12_lstm_3_lstm_cell_3_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_13AssignVariableOp7assignvariableop_13_lstm_3_lstm_cell_3_recurrent_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_14AssignVariableOp+assignvariableop_14_lstm_3_lstm_cell_3_biasIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_15AssignVariableOpassignvariableop_15_totalIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_16AssignVariableOpassignvariableop_16_countIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_17AssignVariableOpassignvariableop_17_total_1Identity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_18AssignVariableOpassignvariableop_18_count_1Identity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_19AssignVariableOp)assignvariableop_19_adam_dense_2_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_20AssignVariableOp'assignvariableop_20_adam_dense_2_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_21AssignVariableOp)assignvariableop_21_adam_dense_3_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_22AssignVariableOp'assignvariableop_22_adam_dense_3_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_23AssignVariableOp4assignvariableop_23_adam_lstm_2_lstm_cell_2_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_24AssignVariableOp>assignvariableop_24_adam_lstm_2_lstm_cell_2_recurrent_kernel_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_25AssignVariableOp2assignvariableop_25_adam_lstm_2_lstm_cell_2_bias_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_26AssignVariableOp4assignvariableop_26_adam_lstm_3_lstm_cell_3_kernel_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_27AssignVariableOp>assignvariableop_27_adam_lstm_3_lstm_cell_3_recurrent_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_28AssignVariableOp2assignvariableop_28_adam_lstm_3_lstm_cell_3_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_29AssignVariableOp)assignvariableop_29_adam_dense_2_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_30AssignVariableOp'assignvariableop_30_adam_dense_2_bias_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_31AssignVariableOp)assignvariableop_31_adam_dense_3_kernel_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_32AssignVariableOp'assignvariableop_32_adam_dense_3_bias_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_33AssignVariableOp4assignvariableop_33_adam_lstm_2_lstm_cell_2_kernel_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_34AssignVariableOp>assignvariableop_34_adam_lstm_2_lstm_cell_2_recurrent_kernel_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_35AssignVariableOp2assignvariableop_35_adam_lstm_2_lstm_cell_2_bias_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_36AssignVariableOp4assignvariableop_36_adam_lstm_3_lstm_cell_3_kernel_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_37AssignVariableOp>assignvariableop_37_adam_lstm_3_lstm_cell_3_recurrent_kernel_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_38AssignVariableOp2assignvariableop_38_adam_lstm_3_lstm_cell_3_bias_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?
Identity_39Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_40IdentityIdentity_39:output:0^NoOp_1*
T0*
_output_shapes
: ?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_40Identity_40:output:0*c
_input_shapesR
P: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
?
&__inference_lstm_2_layer_call_fn_58378

inputs
unknown:	?
	unknown_0:	?
	unknown_1:
??
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_lstm_2_layer_call_and_return_conditional_losses_55130t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
lstm_2_while_cond_56864*
&lstm_2_while_lstm_2_while_loop_counter0
,lstm_2_while_lstm_2_while_maximum_iterations
lstm_2_while_placeholder
lstm_2_while_placeholder_1
lstm_2_while_placeholder_2
lstm_2_while_placeholder_3,
(lstm_2_while_less_lstm_2_strided_slice_1A
=lstm_2_while_lstm_2_while_cond_56864___redundant_placeholder0A
=lstm_2_while_lstm_2_while_cond_56864___redundant_placeholder1A
=lstm_2_while_lstm_2_while_cond_56864___redundant_placeholder2A
=lstm_2_while_lstm_2_while_cond_56864___redundant_placeholder3
lstm_2_while_identity
~
lstm_2/while/LessLesslstm_2_while_placeholder(lstm_2_while_less_lstm_2_strided_slice_1*
T0*
_output_shapes
: Y
lstm_2/while/IdentityIdentitylstm_2/while/Less:z:0*
T0
*
_output_shapes
: "7
lstm_2_while_identitylstm_2/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
??
?
 __inference__wrapped_model_53944
lstm_2_inputP
=sequential_1_lstm_2_lstm_cell_2_split_readvariableop_resource:	?N
?sequential_1_lstm_2_lstm_cell_2_split_1_readvariableop_resource:	?K
7sequential_1_lstm_2_lstm_cell_2_readvariableop_resource:
??Q
=sequential_1_lstm_3_lstm_cell_3_split_readvariableop_resource:
??N
?sequential_1_lstm_3_lstm_cell_3_split_1_readvariableop_resource:	?K
7sequential_1_lstm_3_lstm_cell_3_readvariableop_resource:
??G
3sequential_1_dense_2_matmul_readvariableop_resource:
??C
4sequential_1_dense_2_biasadd_readvariableop_resource:	?F
3sequential_1_dense_3_matmul_readvariableop_resource:	?B
4sequential_1_dense_3_biasadd_readvariableop_resource:
identity??+sequential_1/dense_2/BiasAdd/ReadVariableOp?*sequential_1/dense_2/MatMul/ReadVariableOp?+sequential_1/dense_3/BiasAdd/ReadVariableOp?*sequential_1/dense_3/MatMul/ReadVariableOp?.sequential_1/lstm_2/lstm_cell_2/ReadVariableOp?0sequential_1/lstm_2/lstm_cell_2/ReadVariableOp_1?0sequential_1/lstm_2/lstm_cell_2/ReadVariableOp_2?0sequential_1/lstm_2/lstm_cell_2/ReadVariableOp_3?4sequential_1/lstm_2/lstm_cell_2/split/ReadVariableOp?6sequential_1/lstm_2/lstm_cell_2/split_1/ReadVariableOp?sequential_1/lstm_2/while?.sequential_1/lstm_3/lstm_cell_3/ReadVariableOp?0sequential_1/lstm_3/lstm_cell_3/ReadVariableOp_1?0sequential_1/lstm_3/lstm_cell_3/ReadVariableOp_2?0sequential_1/lstm_3/lstm_cell_3/ReadVariableOp_3?4sequential_1/lstm_3/lstm_cell_3/split/ReadVariableOp?6sequential_1/lstm_3/lstm_cell_3/split_1/ReadVariableOp?sequential_1/lstm_3/whileU
sequential_1/lstm_2/ShapeShapelstm_2_input*
T0*
_output_shapes
:q
'sequential_1/lstm_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)sequential_1/lstm_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)sequential_1/lstm_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
!sequential_1/lstm_2/strided_sliceStridedSlice"sequential_1/lstm_2/Shape:output:00sequential_1/lstm_2/strided_slice/stack:output:02sequential_1/lstm_2/strided_slice/stack_1:output:02sequential_1/lstm_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
"sequential_1/lstm_2/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :??
 sequential_1/lstm_2/zeros/packedPack*sequential_1/lstm_2/strided_slice:output:0+sequential_1/lstm_2/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:d
sequential_1/lstm_2/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
sequential_1/lstm_2/zerosFill)sequential_1/lstm_2/zeros/packed:output:0(sequential_1/lstm_2/zeros/Const:output:0*
T0*(
_output_shapes
:??????????g
$sequential_1/lstm_2/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :??
"sequential_1/lstm_2/zeros_1/packedPack*sequential_1/lstm_2/strided_slice:output:0-sequential_1/lstm_2/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:f
!sequential_1/lstm_2/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
sequential_1/lstm_2/zeros_1Fill+sequential_1/lstm_2/zeros_1/packed:output:0*sequential_1/lstm_2/zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????w
"sequential_1/lstm_2/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
sequential_1/lstm_2/transpose	Transposelstm_2_input+sequential_1/lstm_2/transpose/perm:output:0*
T0*+
_output_shapes
:?????????l
sequential_1/lstm_2/Shape_1Shape!sequential_1/lstm_2/transpose:y:0*
T0*
_output_shapes
:s
)sequential_1/lstm_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+sequential_1/lstm_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+sequential_1/lstm_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
#sequential_1/lstm_2/strided_slice_1StridedSlice$sequential_1/lstm_2/Shape_1:output:02sequential_1/lstm_2/strided_slice_1/stack:output:04sequential_1/lstm_2/strided_slice_1/stack_1:output:04sequential_1/lstm_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskz
/sequential_1/lstm_2/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
!sequential_1/lstm_2/TensorArrayV2TensorListReserve8sequential_1/lstm_2/TensorArrayV2/element_shape:output:0,sequential_1/lstm_2/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
Isequential_1/lstm_2/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
;sequential_1/lstm_2/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor!sequential_1/lstm_2/transpose:y:0Rsequential_1/lstm_2/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???s
)sequential_1/lstm_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+sequential_1/lstm_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+sequential_1/lstm_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
#sequential_1/lstm_2/strided_slice_2StridedSlice!sequential_1/lstm_2/transpose:y:02sequential_1/lstm_2/strided_slice_2/stack:output:04sequential_1/lstm_2/strided_slice_2/stack_1:output:04sequential_1/lstm_2/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_maskq
/sequential_1/lstm_2/lstm_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
4sequential_1/lstm_2/lstm_cell_2/split/ReadVariableOpReadVariableOp=sequential_1_lstm_2_lstm_cell_2_split_readvariableop_resource*
_output_shapes
:	?*
dtype0?
%sequential_1/lstm_2/lstm_cell_2/splitSplit8sequential_1/lstm_2/lstm_cell_2/split/split_dim:output:0<sequential_1/lstm_2/lstm_cell_2/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	?:	?:	?:	?*
	num_split?
&sequential_1/lstm_2/lstm_cell_2/MatMulMatMul,sequential_1/lstm_2/strided_slice_2:output:0.sequential_1/lstm_2/lstm_cell_2/split:output:0*
T0*(
_output_shapes
:???????????
(sequential_1/lstm_2/lstm_cell_2/MatMul_1MatMul,sequential_1/lstm_2/strided_slice_2:output:0.sequential_1/lstm_2/lstm_cell_2/split:output:1*
T0*(
_output_shapes
:???????????
(sequential_1/lstm_2/lstm_cell_2/MatMul_2MatMul,sequential_1/lstm_2/strided_slice_2:output:0.sequential_1/lstm_2/lstm_cell_2/split:output:2*
T0*(
_output_shapes
:???????????
(sequential_1/lstm_2/lstm_cell_2/MatMul_3MatMul,sequential_1/lstm_2/strided_slice_2:output:0.sequential_1/lstm_2/lstm_cell_2/split:output:3*
T0*(
_output_shapes
:??????????s
1sequential_1/lstm_2/lstm_cell_2/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
6sequential_1/lstm_2/lstm_cell_2/split_1/ReadVariableOpReadVariableOp?sequential_1_lstm_2_lstm_cell_2_split_1_readvariableop_resource*
_output_shapes	
:?*
dtype0?
'sequential_1/lstm_2/lstm_cell_2/split_1Split:sequential_1/lstm_2/lstm_cell_2/split_1/split_dim:output:0>sequential_1/lstm_2/lstm_cell_2/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split?
'sequential_1/lstm_2/lstm_cell_2/BiasAddBiasAdd0sequential_1/lstm_2/lstm_cell_2/MatMul:product:00sequential_1/lstm_2/lstm_cell_2/split_1:output:0*
T0*(
_output_shapes
:???????????
)sequential_1/lstm_2/lstm_cell_2/BiasAdd_1BiasAdd2sequential_1/lstm_2/lstm_cell_2/MatMul_1:product:00sequential_1/lstm_2/lstm_cell_2/split_1:output:1*
T0*(
_output_shapes
:???????????
)sequential_1/lstm_2/lstm_cell_2/BiasAdd_2BiasAdd2sequential_1/lstm_2/lstm_cell_2/MatMul_2:product:00sequential_1/lstm_2/lstm_cell_2/split_1:output:2*
T0*(
_output_shapes
:???????????
)sequential_1/lstm_2/lstm_cell_2/BiasAdd_3BiasAdd2sequential_1/lstm_2/lstm_cell_2/MatMul_3:product:00sequential_1/lstm_2/lstm_cell_2/split_1:output:3*
T0*(
_output_shapes
:???????????
.sequential_1/lstm_2/lstm_cell_2/ReadVariableOpReadVariableOp7sequential_1_lstm_2_lstm_cell_2_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
3sequential_1/lstm_2/lstm_cell_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
5sequential_1/lstm_2/lstm_cell_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
5sequential_1/lstm_2/lstm_cell_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
-sequential_1/lstm_2/lstm_cell_2/strided_sliceStridedSlice6sequential_1/lstm_2/lstm_cell_2/ReadVariableOp:value:0<sequential_1/lstm_2/lstm_cell_2/strided_slice/stack:output:0>sequential_1/lstm_2/lstm_cell_2/strided_slice/stack_1:output:0>sequential_1/lstm_2/lstm_cell_2/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
(sequential_1/lstm_2/lstm_cell_2/MatMul_4MatMul"sequential_1/lstm_2/zeros:output:06sequential_1/lstm_2/lstm_cell_2/strided_slice:output:0*
T0*(
_output_shapes
:???????????
#sequential_1/lstm_2/lstm_cell_2/addAddV20sequential_1/lstm_2/lstm_cell_2/BiasAdd:output:02sequential_1/lstm_2/lstm_cell_2/MatMul_4:product:0*
T0*(
_output_shapes
:??????????j
%sequential_1/lstm_2/lstm_cell_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>l
'sequential_1/lstm_2/lstm_cell_2/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ??
#sequential_1/lstm_2/lstm_cell_2/MulMul'sequential_1/lstm_2/lstm_cell_2/add:z:0.sequential_1/lstm_2/lstm_cell_2/Const:output:0*
T0*(
_output_shapes
:???????????
%sequential_1/lstm_2/lstm_cell_2/Add_1AddV2'sequential_1/lstm_2/lstm_cell_2/Mul:z:00sequential_1/lstm_2/lstm_cell_2/Const_1:output:0*
T0*(
_output_shapes
:??????????|
7sequential_1/lstm_2/lstm_cell_2/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
5sequential_1/lstm_2/lstm_cell_2/clip_by_value/MinimumMinimum)sequential_1/lstm_2/lstm_cell_2/Add_1:z:0@sequential_1/lstm_2/lstm_cell_2/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:??????????t
/sequential_1/lstm_2/lstm_cell_2/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
-sequential_1/lstm_2/lstm_cell_2/clip_by_valueMaximum9sequential_1/lstm_2/lstm_cell_2/clip_by_value/Minimum:z:08sequential_1/lstm_2/lstm_cell_2/clip_by_value/y:output:0*
T0*(
_output_shapes
:???????????
0sequential_1/lstm_2/lstm_cell_2/ReadVariableOp_1ReadVariableOp7sequential_1_lstm_2_lstm_cell_2_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
5sequential_1/lstm_2/lstm_cell_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       ?
7sequential_1/lstm_2/lstm_cell_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
7sequential_1/lstm_2/lstm_cell_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
/sequential_1/lstm_2/lstm_cell_2/strided_slice_1StridedSlice8sequential_1/lstm_2/lstm_cell_2/ReadVariableOp_1:value:0>sequential_1/lstm_2/lstm_cell_2/strided_slice_1/stack:output:0@sequential_1/lstm_2/lstm_cell_2/strided_slice_1/stack_1:output:0@sequential_1/lstm_2/lstm_cell_2/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
(sequential_1/lstm_2/lstm_cell_2/MatMul_5MatMul"sequential_1/lstm_2/zeros:output:08sequential_1/lstm_2/lstm_cell_2/strided_slice_1:output:0*
T0*(
_output_shapes
:???????????
%sequential_1/lstm_2/lstm_cell_2/add_2AddV22sequential_1/lstm_2/lstm_cell_2/BiasAdd_1:output:02sequential_1/lstm_2/lstm_cell_2/MatMul_5:product:0*
T0*(
_output_shapes
:??????????l
'sequential_1/lstm_2/lstm_cell_2/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>l
'sequential_1/lstm_2/lstm_cell_2/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ??
%sequential_1/lstm_2/lstm_cell_2/Mul_1Mul)sequential_1/lstm_2/lstm_cell_2/add_2:z:00sequential_1/lstm_2/lstm_cell_2/Const_2:output:0*
T0*(
_output_shapes
:???????????
%sequential_1/lstm_2/lstm_cell_2/Add_3AddV2)sequential_1/lstm_2/lstm_cell_2/Mul_1:z:00sequential_1/lstm_2/lstm_cell_2/Const_3:output:0*
T0*(
_output_shapes
:??????????~
9sequential_1/lstm_2/lstm_cell_2/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
7sequential_1/lstm_2/lstm_cell_2/clip_by_value_1/MinimumMinimum)sequential_1/lstm_2/lstm_cell_2/Add_3:z:0Bsequential_1/lstm_2/lstm_cell_2/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:??????????v
1sequential_1/lstm_2/lstm_cell_2/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
/sequential_1/lstm_2/lstm_cell_2/clip_by_value_1Maximum;sequential_1/lstm_2/lstm_cell_2/clip_by_value_1/Minimum:z:0:sequential_1/lstm_2/lstm_cell_2/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:???????????
%sequential_1/lstm_2/lstm_cell_2/mul_2Mul3sequential_1/lstm_2/lstm_cell_2/clip_by_value_1:z:0$sequential_1/lstm_2/zeros_1:output:0*
T0*(
_output_shapes
:???????????
0sequential_1/lstm_2/lstm_cell_2/ReadVariableOp_2ReadVariableOp7sequential_1_lstm_2_lstm_cell_2_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
5sequential_1/lstm_2/lstm_cell_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       ?
7sequential_1/lstm_2/lstm_cell_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
7sequential_1/lstm_2/lstm_cell_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
/sequential_1/lstm_2/lstm_cell_2/strided_slice_2StridedSlice8sequential_1/lstm_2/lstm_cell_2/ReadVariableOp_2:value:0>sequential_1/lstm_2/lstm_cell_2/strided_slice_2/stack:output:0@sequential_1/lstm_2/lstm_cell_2/strided_slice_2/stack_1:output:0@sequential_1/lstm_2/lstm_cell_2/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
(sequential_1/lstm_2/lstm_cell_2/MatMul_6MatMul"sequential_1/lstm_2/zeros:output:08sequential_1/lstm_2/lstm_cell_2/strided_slice_2:output:0*
T0*(
_output_shapes
:???????????
%sequential_1/lstm_2/lstm_cell_2/add_4AddV22sequential_1/lstm_2/lstm_cell_2/BiasAdd_2:output:02sequential_1/lstm_2/lstm_cell_2/MatMul_6:product:0*
T0*(
_output_shapes
:???????????
$sequential_1/lstm_2/lstm_cell_2/TanhTanh)sequential_1/lstm_2/lstm_cell_2/add_4:z:0*
T0*(
_output_shapes
:???????????
%sequential_1/lstm_2/lstm_cell_2/mul_3Mul1sequential_1/lstm_2/lstm_cell_2/clip_by_value:z:0(sequential_1/lstm_2/lstm_cell_2/Tanh:y:0*
T0*(
_output_shapes
:???????????
%sequential_1/lstm_2/lstm_cell_2/add_5AddV2)sequential_1/lstm_2/lstm_cell_2/mul_2:z:0)sequential_1/lstm_2/lstm_cell_2/mul_3:z:0*
T0*(
_output_shapes
:???????????
0sequential_1/lstm_2/lstm_cell_2/ReadVariableOp_3ReadVariableOp7sequential_1_lstm_2_lstm_cell_2_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
5sequential_1/lstm_2/lstm_cell_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       ?
7sequential_1/lstm_2/lstm_cell_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        ?
7sequential_1/lstm_2/lstm_cell_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
/sequential_1/lstm_2/lstm_cell_2/strided_slice_3StridedSlice8sequential_1/lstm_2/lstm_cell_2/ReadVariableOp_3:value:0>sequential_1/lstm_2/lstm_cell_2/strided_slice_3/stack:output:0@sequential_1/lstm_2/lstm_cell_2/strided_slice_3/stack_1:output:0@sequential_1/lstm_2/lstm_cell_2/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
(sequential_1/lstm_2/lstm_cell_2/MatMul_7MatMul"sequential_1/lstm_2/zeros:output:08sequential_1/lstm_2/lstm_cell_2/strided_slice_3:output:0*
T0*(
_output_shapes
:???????????
%sequential_1/lstm_2/lstm_cell_2/add_6AddV22sequential_1/lstm_2/lstm_cell_2/BiasAdd_3:output:02sequential_1/lstm_2/lstm_cell_2/MatMul_7:product:0*
T0*(
_output_shapes
:??????????l
'sequential_1/lstm_2/lstm_cell_2/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>l
'sequential_1/lstm_2/lstm_cell_2/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ??
%sequential_1/lstm_2/lstm_cell_2/Mul_4Mul)sequential_1/lstm_2/lstm_cell_2/add_6:z:00sequential_1/lstm_2/lstm_cell_2/Const_4:output:0*
T0*(
_output_shapes
:???????????
%sequential_1/lstm_2/lstm_cell_2/Add_7AddV2)sequential_1/lstm_2/lstm_cell_2/Mul_4:z:00sequential_1/lstm_2/lstm_cell_2/Const_5:output:0*
T0*(
_output_shapes
:??????????~
9sequential_1/lstm_2/lstm_cell_2/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
7sequential_1/lstm_2/lstm_cell_2/clip_by_value_2/MinimumMinimum)sequential_1/lstm_2/lstm_cell_2/Add_7:z:0Bsequential_1/lstm_2/lstm_cell_2/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:??????????v
1sequential_1/lstm_2/lstm_cell_2/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
/sequential_1/lstm_2/lstm_cell_2/clip_by_value_2Maximum;sequential_1/lstm_2/lstm_cell_2/clip_by_value_2/Minimum:z:0:sequential_1/lstm_2/lstm_cell_2/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:???????????
&sequential_1/lstm_2/lstm_cell_2/Tanh_1Tanh)sequential_1/lstm_2/lstm_cell_2/add_5:z:0*
T0*(
_output_shapes
:???????????
%sequential_1/lstm_2/lstm_cell_2/mul_5Mul3sequential_1/lstm_2/lstm_cell_2/clip_by_value_2:z:0*sequential_1/lstm_2/lstm_cell_2/Tanh_1:y:0*
T0*(
_output_shapes
:???????????
1sequential_1/lstm_2/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
#sequential_1/lstm_2/TensorArrayV2_1TensorListReserve:sequential_1/lstm_2/TensorArrayV2_1/element_shape:output:0,sequential_1/lstm_2/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???Z
sequential_1/lstm_2/timeConst*
_output_shapes
: *
dtype0*
value	B : w
,sequential_1/lstm_2/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????h
&sequential_1/lstm_2/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
sequential_1/lstm_2/whileWhile/sequential_1/lstm_2/while/loop_counter:output:05sequential_1/lstm_2/while/maximum_iterations:output:0!sequential_1/lstm_2/time:output:0,sequential_1/lstm_2/TensorArrayV2_1:handle:0"sequential_1/lstm_2/zeros:output:0$sequential_1/lstm_2/zeros_1:output:0,sequential_1/lstm_2/strided_slice_1:output:0Ksequential_1/lstm_2/TensorArrayUnstack/TensorListFromTensor:output_handle:0=sequential_1_lstm_2_lstm_cell_2_split_readvariableop_resource?sequential_1_lstm_2_lstm_cell_2_split_1_readvariableop_resource7sequential_1_lstm_2_lstm_cell_2_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *0
body(R&
$sequential_1_lstm_2_while_body_53538*0
cond(R&
$sequential_1_lstm_2_while_cond_53537*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations ?
Dsequential_1/lstm_2/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
6sequential_1/lstm_2/TensorArrayV2Stack/TensorListStackTensorListStack"sequential_1/lstm_2/while:output:3Msequential_1/lstm_2/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
element_dtype0|
)sequential_1/lstm_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????u
+sequential_1/lstm_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: u
+sequential_1/lstm_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
#sequential_1/lstm_2/strided_slice_3StridedSlice?sequential_1/lstm_2/TensorArrayV2Stack/TensorListStack:tensor:02sequential_1/lstm_2/strided_slice_3/stack:output:04sequential_1/lstm_2/strided_slice_3/stack_1:output:04sequential_1/lstm_2/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_masky
$sequential_1/lstm_2/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
sequential_1/lstm_2/transpose_1	Transpose?sequential_1/lstm_2/TensorArrayV2Stack/TensorListStack:tensor:0-sequential_1/lstm_2/transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????l
sequential_1/lstm_3/ShapeShape#sequential_1/lstm_2/transpose_1:y:0*
T0*
_output_shapes
:q
'sequential_1/lstm_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)sequential_1/lstm_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)sequential_1/lstm_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
!sequential_1/lstm_3/strided_sliceStridedSlice"sequential_1/lstm_3/Shape:output:00sequential_1/lstm_3/strided_slice/stack:output:02sequential_1/lstm_3/strided_slice/stack_1:output:02sequential_1/lstm_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
"sequential_1/lstm_3/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :??
 sequential_1/lstm_3/zeros/packedPack*sequential_1/lstm_3/strided_slice:output:0+sequential_1/lstm_3/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:d
sequential_1/lstm_3/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
sequential_1/lstm_3/zerosFill)sequential_1/lstm_3/zeros/packed:output:0(sequential_1/lstm_3/zeros/Const:output:0*
T0*(
_output_shapes
:??????????g
$sequential_1/lstm_3/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :??
"sequential_1/lstm_3/zeros_1/packedPack*sequential_1/lstm_3/strided_slice:output:0-sequential_1/lstm_3/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:f
!sequential_1/lstm_3/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
sequential_1/lstm_3/zeros_1Fill+sequential_1/lstm_3/zeros_1/packed:output:0*sequential_1/lstm_3/zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????w
"sequential_1/lstm_3/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
sequential_1/lstm_3/transpose	Transpose#sequential_1/lstm_2/transpose_1:y:0+sequential_1/lstm_3/transpose/perm:output:0*
T0*,
_output_shapes
:??????????l
sequential_1/lstm_3/Shape_1Shape!sequential_1/lstm_3/transpose:y:0*
T0*
_output_shapes
:s
)sequential_1/lstm_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+sequential_1/lstm_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+sequential_1/lstm_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
#sequential_1/lstm_3/strided_slice_1StridedSlice$sequential_1/lstm_3/Shape_1:output:02sequential_1/lstm_3/strided_slice_1/stack:output:04sequential_1/lstm_3/strided_slice_1/stack_1:output:04sequential_1/lstm_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskz
/sequential_1/lstm_3/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
!sequential_1/lstm_3/TensorArrayV2TensorListReserve8sequential_1/lstm_3/TensorArrayV2/element_shape:output:0,sequential_1/lstm_3/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
Isequential_1/lstm_3/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
;sequential_1/lstm_3/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor!sequential_1/lstm_3/transpose:y:0Rsequential_1/lstm_3/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???s
)sequential_1/lstm_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+sequential_1/lstm_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+sequential_1/lstm_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
#sequential_1/lstm_3/strided_slice_2StridedSlice!sequential_1/lstm_3/transpose:y:02sequential_1/lstm_3/strided_slice_2/stack:output:04sequential_1/lstm_3/strided_slice_2/stack_1:output:04sequential_1/lstm_3/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_maskq
/sequential_1/lstm_3/lstm_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
4sequential_1/lstm_3/lstm_cell_3/split/ReadVariableOpReadVariableOp=sequential_1_lstm_3_lstm_cell_3_split_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
%sequential_1/lstm_3/lstm_cell_3/splitSplit8sequential_1/lstm_3/lstm_cell_3/split/split_dim:output:0<sequential_1/lstm_3/lstm_cell_3/split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
??:
??:
??:
??*
	num_split?
&sequential_1/lstm_3/lstm_cell_3/MatMulMatMul,sequential_1/lstm_3/strided_slice_2:output:0.sequential_1/lstm_3/lstm_cell_3/split:output:0*
T0*(
_output_shapes
:???????????
(sequential_1/lstm_3/lstm_cell_3/MatMul_1MatMul,sequential_1/lstm_3/strided_slice_2:output:0.sequential_1/lstm_3/lstm_cell_3/split:output:1*
T0*(
_output_shapes
:???????????
(sequential_1/lstm_3/lstm_cell_3/MatMul_2MatMul,sequential_1/lstm_3/strided_slice_2:output:0.sequential_1/lstm_3/lstm_cell_3/split:output:2*
T0*(
_output_shapes
:???????????
(sequential_1/lstm_3/lstm_cell_3/MatMul_3MatMul,sequential_1/lstm_3/strided_slice_2:output:0.sequential_1/lstm_3/lstm_cell_3/split:output:3*
T0*(
_output_shapes
:??????????s
1sequential_1/lstm_3/lstm_cell_3/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
6sequential_1/lstm_3/lstm_cell_3/split_1/ReadVariableOpReadVariableOp?sequential_1_lstm_3_lstm_cell_3_split_1_readvariableop_resource*
_output_shapes	
:?*
dtype0?
'sequential_1/lstm_3/lstm_cell_3/split_1Split:sequential_1/lstm_3/lstm_cell_3/split_1/split_dim:output:0>sequential_1/lstm_3/lstm_cell_3/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split?
'sequential_1/lstm_3/lstm_cell_3/BiasAddBiasAdd0sequential_1/lstm_3/lstm_cell_3/MatMul:product:00sequential_1/lstm_3/lstm_cell_3/split_1:output:0*
T0*(
_output_shapes
:???????????
)sequential_1/lstm_3/lstm_cell_3/BiasAdd_1BiasAdd2sequential_1/lstm_3/lstm_cell_3/MatMul_1:product:00sequential_1/lstm_3/lstm_cell_3/split_1:output:1*
T0*(
_output_shapes
:???????????
)sequential_1/lstm_3/lstm_cell_3/BiasAdd_2BiasAdd2sequential_1/lstm_3/lstm_cell_3/MatMul_2:product:00sequential_1/lstm_3/lstm_cell_3/split_1:output:2*
T0*(
_output_shapes
:???????????
)sequential_1/lstm_3/lstm_cell_3/BiasAdd_3BiasAdd2sequential_1/lstm_3/lstm_cell_3/MatMul_3:product:00sequential_1/lstm_3/lstm_cell_3/split_1:output:3*
T0*(
_output_shapes
:???????????
.sequential_1/lstm_3/lstm_cell_3/ReadVariableOpReadVariableOp7sequential_1_lstm_3_lstm_cell_3_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
3sequential_1/lstm_3/lstm_cell_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
5sequential_1/lstm_3/lstm_cell_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   ?
5sequential_1/lstm_3/lstm_cell_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
-sequential_1/lstm_3/lstm_cell_3/strided_sliceStridedSlice6sequential_1/lstm_3/lstm_cell_3/ReadVariableOp:value:0<sequential_1/lstm_3/lstm_cell_3/strided_slice/stack:output:0>sequential_1/lstm_3/lstm_cell_3/strided_slice/stack_1:output:0>sequential_1/lstm_3/lstm_cell_3/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
(sequential_1/lstm_3/lstm_cell_3/MatMul_4MatMul"sequential_1/lstm_3/zeros:output:06sequential_1/lstm_3/lstm_cell_3/strided_slice:output:0*
T0*(
_output_shapes
:???????????
#sequential_1/lstm_3/lstm_cell_3/addAddV20sequential_1/lstm_3/lstm_cell_3/BiasAdd:output:02sequential_1/lstm_3/lstm_cell_3/MatMul_4:product:0*
T0*(
_output_shapes
:??????????j
%sequential_1/lstm_3/lstm_cell_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>l
'sequential_1/lstm_3/lstm_cell_3/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ??
#sequential_1/lstm_3/lstm_cell_3/MulMul'sequential_1/lstm_3/lstm_cell_3/add:z:0.sequential_1/lstm_3/lstm_cell_3/Const:output:0*
T0*(
_output_shapes
:???????????
%sequential_1/lstm_3/lstm_cell_3/Add_1AddV2'sequential_1/lstm_3/lstm_cell_3/Mul:z:00sequential_1/lstm_3/lstm_cell_3/Const_1:output:0*
T0*(
_output_shapes
:??????????|
7sequential_1/lstm_3/lstm_cell_3/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
5sequential_1/lstm_3/lstm_cell_3/clip_by_value/MinimumMinimum)sequential_1/lstm_3/lstm_cell_3/Add_1:z:0@sequential_1/lstm_3/lstm_cell_3/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:??????????t
/sequential_1/lstm_3/lstm_cell_3/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
-sequential_1/lstm_3/lstm_cell_3/clip_by_valueMaximum9sequential_1/lstm_3/lstm_cell_3/clip_by_value/Minimum:z:08sequential_1/lstm_3/lstm_cell_3/clip_by_value/y:output:0*
T0*(
_output_shapes
:???????????
0sequential_1/lstm_3/lstm_cell_3/ReadVariableOp_1ReadVariableOp7sequential_1_lstm_3_lstm_cell_3_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
5sequential_1/lstm_3/lstm_cell_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   ?
7sequential_1/lstm_3/lstm_cell_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
7sequential_1/lstm_3/lstm_cell_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
/sequential_1/lstm_3/lstm_cell_3/strided_slice_1StridedSlice8sequential_1/lstm_3/lstm_cell_3/ReadVariableOp_1:value:0>sequential_1/lstm_3/lstm_cell_3/strided_slice_1/stack:output:0@sequential_1/lstm_3/lstm_cell_3/strided_slice_1/stack_1:output:0@sequential_1/lstm_3/lstm_cell_3/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
(sequential_1/lstm_3/lstm_cell_3/MatMul_5MatMul"sequential_1/lstm_3/zeros:output:08sequential_1/lstm_3/lstm_cell_3/strided_slice_1:output:0*
T0*(
_output_shapes
:???????????
%sequential_1/lstm_3/lstm_cell_3/add_2AddV22sequential_1/lstm_3/lstm_cell_3/BiasAdd_1:output:02sequential_1/lstm_3/lstm_cell_3/MatMul_5:product:0*
T0*(
_output_shapes
:??????????l
'sequential_1/lstm_3/lstm_cell_3/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>l
'sequential_1/lstm_3/lstm_cell_3/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ??
%sequential_1/lstm_3/lstm_cell_3/Mul_1Mul)sequential_1/lstm_3/lstm_cell_3/add_2:z:00sequential_1/lstm_3/lstm_cell_3/Const_2:output:0*
T0*(
_output_shapes
:???????????
%sequential_1/lstm_3/lstm_cell_3/Add_3AddV2)sequential_1/lstm_3/lstm_cell_3/Mul_1:z:00sequential_1/lstm_3/lstm_cell_3/Const_3:output:0*
T0*(
_output_shapes
:??????????~
9sequential_1/lstm_3/lstm_cell_3/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
7sequential_1/lstm_3/lstm_cell_3/clip_by_value_1/MinimumMinimum)sequential_1/lstm_3/lstm_cell_3/Add_3:z:0Bsequential_1/lstm_3/lstm_cell_3/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:??????????v
1sequential_1/lstm_3/lstm_cell_3/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
/sequential_1/lstm_3/lstm_cell_3/clip_by_value_1Maximum;sequential_1/lstm_3/lstm_cell_3/clip_by_value_1/Minimum:z:0:sequential_1/lstm_3/lstm_cell_3/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:???????????
%sequential_1/lstm_3/lstm_cell_3/mul_2Mul3sequential_1/lstm_3/lstm_cell_3/clip_by_value_1:z:0$sequential_1/lstm_3/zeros_1:output:0*
T0*(
_output_shapes
:???????????
0sequential_1/lstm_3/lstm_cell_3/ReadVariableOp_2ReadVariableOp7sequential_1_lstm_3_lstm_cell_3_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
5sequential_1/lstm_3/lstm_cell_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       ?
7sequential_1/lstm_3/lstm_cell_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  ?
7sequential_1/lstm_3/lstm_cell_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
/sequential_1/lstm_3/lstm_cell_3/strided_slice_2StridedSlice8sequential_1/lstm_3/lstm_cell_3/ReadVariableOp_2:value:0>sequential_1/lstm_3/lstm_cell_3/strided_slice_2/stack:output:0@sequential_1/lstm_3/lstm_cell_3/strided_slice_2/stack_1:output:0@sequential_1/lstm_3/lstm_cell_3/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
(sequential_1/lstm_3/lstm_cell_3/MatMul_6MatMul"sequential_1/lstm_3/zeros:output:08sequential_1/lstm_3/lstm_cell_3/strided_slice_2:output:0*
T0*(
_output_shapes
:???????????
%sequential_1/lstm_3/lstm_cell_3/add_4AddV22sequential_1/lstm_3/lstm_cell_3/BiasAdd_2:output:02sequential_1/lstm_3/lstm_cell_3/MatMul_6:product:0*
T0*(
_output_shapes
:???????????
$sequential_1/lstm_3/lstm_cell_3/TanhTanh)sequential_1/lstm_3/lstm_cell_3/add_4:z:0*
T0*(
_output_shapes
:???????????
%sequential_1/lstm_3/lstm_cell_3/mul_3Mul1sequential_1/lstm_3/lstm_cell_3/clip_by_value:z:0(sequential_1/lstm_3/lstm_cell_3/Tanh:y:0*
T0*(
_output_shapes
:???????????
%sequential_1/lstm_3/lstm_cell_3/add_5AddV2)sequential_1/lstm_3/lstm_cell_3/mul_2:z:0)sequential_1/lstm_3/lstm_cell_3/mul_3:z:0*
T0*(
_output_shapes
:???????????
0sequential_1/lstm_3/lstm_cell_3/ReadVariableOp_3ReadVariableOp7sequential_1_lstm_3_lstm_cell_3_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
5sequential_1/lstm_3/lstm_cell_3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  ?
7sequential_1/lstm_3/lstm_cell_3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        ?
7sequential_1/lstm_3/lstm_cell_3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
/sequential_1/lstm_3/lstm_cell_3/strided_slice_3StridedSlice8sequential_1/lstm_3/lstm_cell_3/ReadVariableOp_3:value:0>sequential_1/lstm_3/lstm_cell_3/strided_slice_3/stack:output:0@sequential_1/lstm_3/lstm_cell_3/strided_slice_3/stack_1:output:0@sequential_1/lstm_3/lstm_cell_3/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
(sequential_1/lstm_3/lstm_cell_3/MatMul_7MatMul"sequential_1/lstm_3/zeros:output:08sequential_1/lstm_3/lstm_cell_3/strided_slice_3:output:0*
T0*(
_output_shapes
:???????????
%sequential_1/lstm_3/lstm_cell_3/add_6AddV22sequential_1/lstm_3/lstm_cell_3/BiasAdd_3:output:02sequential_1/lstm_3/lstm_cell_3/MatMul_7:product:0*
T0*(
_output_shapes
:??????????l
'sequential_1/lstm_3/lstm_cell_3/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>l
'sequential_1/lstm_3/lstm_cell_3/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ??
%sequential_1/lstm_3/lstm_cell_3/Mul_4Mul)sequential_1/lstm_3/lstm_cell_3/add_6:z:00sequential_1/lstm_3/lstm_cell_3/Const_4:output:0*
T0*(
_output_shapes
:???????????
%sequential_1/lstm_3/lstm_cell_3/Add_7AddV2)sequential_1/lstm_3/lstm_cell_3/Mul_4:z:00sequential_1/lstm_3/lstm_cell_3/Const_5:output:0*
T0*(
_output_shapes
:??????????~
9sequential_1/lstm_3/lstm_cell_3/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
7sequential_1/lstm_3/lstm_cell_3/clip_by_value_2/MinimumMinimum)sequential_1/lstm_3/lstm_cell_3/Add_7:z:0Bsequential_1/lstm_3/lstm_cell_3/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:??????????v
1sequential_1/lstm_3/lstm_cell_3/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
/sequential_1/lstm_3/lstm_cell_3/clip_by_value_2Maximum;sequential_1/lstm_3/lstm_cell_3/clip_by_value_2/Minimum:z:0:sequential_1/lstm_3/lstm_cell_3/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:???????????
&sequential_1/lstm_3/lstm_cell_3/Tanh_1Tanh)sequential_1/lstm_3/lstm_cell_3/add_5:z:0*
T0*(
_output_shapes
:???????????
%sequential_1/lstm_3/lstm_cell_3/mul_5Mul3sequential_1/lstm_3/lstm_cell_3/clip_by_value_2:z:0*sequential_1/lstm_3/lstm_cell_3/Tanh_1:y:0*
T0*(
_output_shapes
:???????????
1sequential_1/lstm_3/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   ?
#sequential_1/lstm_3/TensorArrayV2_1TensorListReserve:sequential_1/lstm_3/TensorArrayV2_1/element_shape:output:0,sequential_1/lstm_3/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???Z
sequential_1/lstm_3/timeConst*
_output_shapes
: *
dtype0*
value	B : w
,sequential_1/lstm_3/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????h
&sequential_1/lstm_3/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
sequential_1/lstm_3/whileWhile/sequential_1/lstm_3/while/loop_counter:output:05sequential_1/lstm_3/while/maximum_iterations:output:0!sequential_1/lstm_3/time:output:0,sequential_1/lstm_3/TensorArrayV2_1:handle:0"sequential_1/lstm_3/zeros:output:0$sequential_1/lstm_3/zeros_1:output:0,sequential_1/lstm_3/strided_slice_1:output:0Ksequential_1/lstm_3/TensorArrayUnstack/TensorListFromTensor:output_handle:0=sequential_1_lstm_3_lstm_cell_3_split_readvariableop_resource?sequential_1_lstm_3_lstm_cell_3_split_1_readvariableop_resource7sequential_1_lstm_3_lstm_cell_3_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *0
body(R&
$sequential_1_lstm_3_while_body_53790*0
cond(R&
$sequential_1_lstm_3_while_cond_53789*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations ?
Dsequential_1/lstm_3/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   ?
6sequential_1/lstm_3/TensorArrayV2Stack/TensorListStackTensorListStack"sequential_1/lstm_3/while:output:3Msequential_1/lstm_3/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
element_dtype0|
)sequential_1/lstm_3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????u
+sequential_1/lstm_3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: u
+sequential_1/lstm_3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
#sequential_1/lstm_3/strided_slice_3StridedSlice?sequential_1/lstm_3/TensorArrayV2Stack/TensorListStack:tensor:02sequential_1/lstm_3/strided_slice_3/stack:output:04sequential_1/lstm_3/strided_slice_3/stack_1:output:04sequential_1/lstm_3/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_masky
$sequential_1/lstm_3/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
sequential_1/lstm_3/transpose_1	Transpose?sequential_1/lstm_3/TensorArrayV2Stack/TensorListStack:tensor:0-sequential_1/lstm_3/transpose_1/perm:output:0*
T0*,
_output_shapes
:???????????
*sequential_1/dense_2/MatMul/ReadVariableOpReadVariableOp3sequential_1_dense_2_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
sequential_1/dense_2/MatMulMatMul,sequential_1/lstm_3/strided_slice_3:output:02sequential_1/dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
+sequential_1/dense_2/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
sequential_1/dense_2/BiasAddBiasAdd%sequential_1/dense_2/MatMul:product:03sequential_1/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????{
sequential_1/dense_2/ReluRelu%sequential_1/dense_2/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
*sequential_1/dense_3/MatMul/ReadVariableOpReadVariableOp3sequential_1_dense_3_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
sequential_1/dense_3/MatMulMatMul'sequential_1/dense_2/Relu:activations:02sequential_1/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
+sequential_1/dense_3/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
sequential_1/dense_3/BiasAddBiasAdd%sequential_1/dense_3/MatMul:product:03sequential_1/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
sequential_1/dense_3/SigmoidSigmoid%sequential_1/dense_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????o
IdentityIdentity sequential_1/dense_3/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp,^sequential_1/dense_2/BiasAdd/ReadVariableOp+^sequential_1/dense_2/MatMul/ReadVariableOp,^sequential_1/dense_3/BiasAdd/ReadVariableOp+^sequential_1/dense_3/MatMul/ReadVariableOp/^sequential_1/lstm_2/lstm_cell_2/ReadVariableOp1^sequential_1/lstm_2/lstm_cell_2/ReadVariableOp_11^sequential_1/lstm_2/lstm_cell_2/ReadVariableOp_21^sequential_1/lstm_2/lstm_cell_2/ReadVariableOp_35^sequential_1/lstm_2/lstm_cell_2/split/ReadVariableOp7^sequential_1/lstm_2/lstm_cell_2/split_1/ReadVariableOp^sequential_1/lstm_2/while/^sequential_1/lstm_3/lstm_cell_3/ReadVariableOp1^sequential_1/lstm_3/lstm_cell_3/ReadVariableOp_11^sequential_1/lstm_3/lstm_cell_3/ReadVariableOp_21^sequential_1/lstm_3/lstm_cell_3/ReadVariableOp_35^sequential_1/lstm_3/lstm_cell_3/split/ReadVariableOp7^sequential_1/lstm_3/lstm_cell_3/split_1/ReadVariableOp^sequential_1/lstm_3/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????: : : : : : : : : : 2Z
+sequential_1/dense_2/BiasAdd/ReadVariableOp+sequential_1/dense_2/BiasAdd/ReadVariableOp2X
*sequential_1/dense_2/MatMul/ReadVariableOp*sequential_1/dense_2/MatMul/ReadVariableOp2Z
+sequential_1/dense_3/BiasAdd/ReadVariableOp+sequential_1/dense_3/BiasAdd/ReadVariableOp2X
*sequential_1/dense_3/MatMul/ReadVariableOp*sequential_1/dense_3/MatMul/ReadVariableOp2`
.sequential_1/lstm_2/lstm_cell_2/ReadVariableOp.sequential_1/lstm_2/lstm_cell_2/ReadVariableOp2d
0sequential_1/lstm_2/lstm_cell_2/ReadVariableOp_10sequential_1/lstm_2/lstm_cell_2/ReadVariableOp_12d
0sequential_1/lstm_2/lstm_cell_2/ReadVariableOp_20sequential_1/lstm_2/lstm_cell_2/ReadVariableOp_22d
0sequential_1/lstm_2/lstm_cell_2/ReadVariableOp_30sequential_1/lstm_2/lstm_cell_2/ReadVariableOp_32l
4sequential_1/lstm_2/lstm_cell_2/split/ReadVariableOp4sequential_1/lstm_2/lstm_cell_2/split/ReadVariableOp2p
6sequential_1/lstm_2/lstm_cell_2/split_1/ReadVariableOp6sequential_1/lstm_2/lstm_cell_2/split_1/ReadVariableOp26
sequential_1/lstm_2/whilesequential_1/lstm_2/while2`
.sequential_1/lstm_3/lstm_cell_3/ReadVariableOp.sequential_1/lstm_3/lstm_cell_3/ReadVariableOp2d
0sequential_1/lstm_3/lstm_cell_3/ReadVariableOp_10sequential_1/lstm_3/lstm_cell_3/ReadVariableOp_12d
0sequential_1/lstm_3/lstm_cell_3/ReadVariableOp_20sequential_1/lstm_3/lstm_cell_3/ReadVariableOp_22d
0sequential_1/lstm_3/lstm_cell_3/ReadVariableOp_30sequential_1/lstm_3/lstm_cell_3/ReadVariableOp_32l
4sequential_1/lstm_3/lstm_cell_3/split/ReadVariableOp4sequential_1/lstm_3/lstm_cell_3/split/ReadVariableOp2p
6sequential_1/lstm_3/lstm_cell_3/split_1/ReadVariableOp6sequential_1/lstm_3/lstm_cell_3/split_1/ReadVariableOp26
sequential_1/lstm_3/whilesequential_1/lstm_3/while:Y U
+
_output_shapes
:?????????
&
_user_specified_namelstm_2_input
?
?
while_cond_57692
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_57692___redundant_placeholder03
/while_while_cond_57692___redundant_placeholder13
/while_while_cond_57692___redundant_placeholder23
/while_while_cond_57692___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?
?
'__inference_dense_3_layer_call_fn_59497

inputs
unknown:	?
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_55429o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?
A__inference_lstm_3_layer_call_and_return_conditional_losses_58645
inputs_0=
)lstm_cell_3_split_readvariableop_resource:
??:
+lstm_cell_3_split_1_readvariableop_resource:	?7
#lstm_cell_3_readvariableop_resource:
??
identity??lstm_cell_3/ReadVariableOp?lstm_cell_3/ReadVariableOp_1?lstm_cell_3/ReadVariableOp_2?lstm_cell_3/ReadVariableOp_3? lstm_cell_3/split/ReadVariableOp?"lstm_cell_3/split_1/ReadVariableOp?while=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????S
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          y
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*5
_output_shapes#
!:???????????????????D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask]
lstm_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
 lstm_cell_3/split/ReadVariableOpReadVariableOp)lstm_cell_3_split_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
lstm_cell_3/splitSplit$lstm_cell_3/split/split_dim:output:0(lstm_cell_3/split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
??:
??:
??:
??*
	num_split?
lstm_cell_3/MatMulMatMulstrided_slice_2:output:0lstm_cell_3/split:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_3/MatMul_1MatMulstrided_slice_2:output:0lstm_cell_3/split:output:1*
T0*(
_output_shapes
:???????????
lstm_cell_3/MatMul_2MatMulstrided_slice_2:output:0lstm_cell_3/split:output:2*
T0*(
_output_shapes
:???????????
lstm_cell_3/MatMul_3MatMulstrided_slice_2:output:0lstm_cell_3/split:output:3*
T0*(
_output_shapes
:??????????_
lstm_cell_3/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
"lstm_cell_3/split_1/ReadVariableOpReadVariableOp+lstm_cell_3_split_1_readvariableop_resource*
_output_shapes	
:?*
dtype0?
lstm_cell_3/split_1Split&lstm_cell_3/split_1/split_dim:output:0*lstm_cell_3/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split?
lstm_cell_3/BiasAddBiasAddlstm_cell_3/MatMul:product:0lstm_cell_3/split_1:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_3/BiasAdd_1BiasAddlstm_cell_3/MatMul_1:product:0lstm_cell_3/split_1:output:1*
T0*(
_output_shapes
:???????????
lstm_cell_3/BiasAdd_2BiasAddlstm_cell_3/MatMul_2:product:0lstm_cell_3/split_1:output:2*
T0*(
_output_shapes
:???????????
lstm_cell_3/BiasAdd_3BiasAddlstm_cell_3/MatMul_3:product:0lstm_cell_3/split_1:output:3*
T0*(
_output_shapes
:???????????
lstm_cell_3/ReadVariableOpReadVariableOp#lstm_cell_3_readvariableop_resource* 
_output_shapes
:
??*
dtype0p
lstm_cell_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        r
!lstm_cell_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   r
!lstm_cell_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
lstm_cell_3/strided_sliceStridedSlice"lstm_cell_3/ReadVariableOp:value:0(lstm_cell_3/strided_slice/stack:output:0*lstm_cell_3/strided_slice/stack_1:output:0*lstm_cell_3/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
lstm_cell_3/MatMul_4MatMulzeros:output:0"lstm_cell_3/strided_slice:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_3/addAddV2lstm_cell_3/BiasAdd:output:0lstm_cell_3/MatMul_4:product:0*
T0*(
_output_shapes
:??????????V
lstm_cell_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>X
lstm_cell_3/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?z
lstm_cell_3/MulMullstm_cell_3/add:z:0lstm_cell_3/Const:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_3/Add_1AddV2lstm_cell_3/Mul:z:0lstm_cell_3/Const_1:output:0*
T0*(
_output_shapes
:??????????h
#lstm_cell_3/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
!lstm_cell_3/clip_by_value/MinimumMinimumlstm_cell_3/Add_1:z:0,lstm_cell_3/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:??????????`
lstm_cell_3/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
lstm_cell_3/clip_by_valueMaximum%lstm_cell_3/clip_by_value/Minimum:z:0$lstm_cell_3/clip_by_value/y:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_3/ReadVariableOp_1ReadVariableOp#lstm_cell_3_readvariableop_resource* 
_output_shapes
:
??*
dtype0r
!lstm_cell_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   t
#lstm_cell_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       t
#lstm_cell_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
lstm_cell_3/strided_slice_1StridedSlice$lstm_cell_3/ReadVariableOp_1:value:0*lstm_cell_3/strided_slice_1/stack:output:0,lstm_cell_3/strided_slice_1/stack_1:output:0,lstm_cell_3/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
lstm_cell_3/MatMul_5MatMulzeros:output:0$lstm_cell_3/strided_slice_1:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_3/add_2AddV2lstm_cell_3/BiasAdd_1:output:0lstm_cell_3/MatMul_5:product:0*
T0*(
_output_shapes
:??????????X
lstm_cell_3/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>X
lstm_cell_3/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ??
lstm_cell_3/Mul_1Mullstm_cell_3/add_2:z:0lstm_cell_3/Const_2:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_3/Add_3AddV2lstm_cell_3/Mul_1:z:0lstm_cell_3/Const_3:output:0*
T0*(
_output_shapes
:??????????j
%lstm_cell_3/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
#lstm_cell_3/clip_by_value_1/MinimumMinimumlstm_cell_3/Add_3:z:0.lstm_cell_3/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:??????????b
lstm_cell_3/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
lstm_cell_3/clip_by_value_1Maximum'lstm_cell_3/clip_by_value_1/Minimum:z:0&lstm_cell_3/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:??????????~
lstm_cell_3/mul_2Mullstm_cell_3/clip_by_value_1:z:0zeros_1:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_3/ReadVariableOp_2ReadVariableOp#lstm_cell_3_readvariableop_resource* 
_output_shapes
:
??*
dtype0r
!lstm_cell_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       t
#lstm_cell_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  t
#lstm_cell_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
lstm_cell_3/strided_slice_2StridedSlice$lstm_cell_3/ReadVariableOp_2:value:0*lstm_cell_3/strided_slice_2/stack:output:0,lstm_cell_3/strided_slice_2/stack_1:output:0,lstm_cell_3/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
lstm_cell_3/MatMul_6MatMulzeros:output:0$lstm_cell_3/strided_slice_2:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_3/add_4AddV2lstm_cell_3/BiasAdd_2:output:0lstm_cell_3/MatMul_6:product:0*
T0*(
_output_shapes
:??????????b
lstm_cell_3/TanhTanhlstm_cell_3/add_4:z:0*
T0*(
_output_shapes
:???????????
lstm_cell_3/mul_3Mullstm_cell_3/clip_by_value:z:0lstm_cell_3/Tanh:y:0*
T0*(
_output_shapes
:??????????{
lstm_cell_3/add_5AddV2lstm_cell_3/mul_2:z:0lstm_cell_3/mul_3:z:0*
T0*(
_output_shapes
:???????????
lstm_cell_3/ReadVariableOp_3ReadVariableOp#lstm_cell_3_readvariableop_resource* 
_output_shapes
:
??*
dtype0r
!lstm_cell_3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  t
#lstm_cell_3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        t
#lstm_cell_3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
lstm_cell_3/strided_slice_3StridedSlice$lstm_cell_3/ReadVariableOp_3:value:0*lstm_cell_3/strided_slice_3/stack:output:0,lstm_cell_3/strided_slice_3/stack_1:output:0,lstm_cell_3/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
lstm_cell_3/MatMul_7MatMulzeros:output:0$lstm_cell_3/strided_slice_3:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_3/add_6AddV2lstm_cell_3/BiasAdd_3:output:0lstm_cell_3/MatMul_7:product:0*
T0*(
_output_shapes
:??????????X
lstm_cell_3/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>X
lstm_cell_3/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ??
lstm_cell_3/Mul_4Mullstm_cell_3/add_6:z:0lstm_cell_3/Const_4:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_3/Add_7AddV2lstm_cell_3/Mul_4:z:0lstm_cell_3/Const_5:output:0*
T0*(
_output_shapes
:??????????j
%lstm_cell_3/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
#lstm_cell_3/clip_by_value_2/MinimumMinimumlstm_cell_3/Add_7:z:0.lstm_cell_3/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:??????????b
lstm_cell_3/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
lstm_cell_3/clip_by_value_2Maximum'lstm_cell_3/clip_by_value_2/Minimum:z:0&lstm_cell_3/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:??????????d
lstm_cell_3/Tanh_1Tanhlstm_cell_3/add_5:z:0*
T0*(
_output_shapes
:???????????
lstm_cell_3/mul_5Mullstm_cell_3/clip_by_value_2:z:0lstm_cell_3/Tanh_1:y:0*
T0*(
_output_shapes
:??????????n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   ?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)lstm_cell_3_split_readvariableop_resource+lstm_cell_3_split_1_readvariableop_resource#lstm_cell_3_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_58505*
condR
while_cond_58504*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:???????????????????*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:???????????????????h
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp^lstm_cell_3/ReadVariableOp^lstm_cell_3/ReadVariableOp_1^lstm_cell_3/ReadVariableOp_2^lstm_cell_3/ReadVariableOp_3!^lstm_cell_3/split/ReadVariableOp#^lstm_cell_3/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':???????????????????: : : 28
lstm_cell_3/ReadVariableOplstm_cell_3/ReadVariableOp2<
lstm_cell_3/ReadVariableOp_1lstm_cell_3/ReadVariableOp_12<
lstm_cell_3/ReadVariableOp_2lstm_cell_3/ReadVariableOp_22<
lstm_cell_3/ReadVariableOp_3lstm_cell_3/ReadVariableOp_32D
 lstm_cell_3/split/ReadVariableOp lstm_cell_3/split/ReadVariableOp2H
"lstm_cell_3/split_1/ReadVariableOp"lstm_cell_3/split_1/ReadVariableOp2
whilewhile:_ [
5
_output_shapes#
!:???????????????????
"
_user_specified_name
inputs/0
?K
?
F__inference_lstm_cell_3_layer_call_and_return_conditional_losses_59798

inputs
states_0
states_11
split_readvariableop_resource:
??.
split_1_readvariableop_resource:	?+
readvariableop_resource:
??
identity

identity_1

identity_2??ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?ReadVariableOp_3?split/ReadVariableOp?split_1/ReadVariableOpQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :t
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
??:
??:
??:
??*
	num_split[
MatMulMatMulinputssplit:output:0*
T0*(
_output_shapes
:??????????]
MatMul_1MatMulinputssplit:output:1*
T0*(
_output_shapes
:??????????]
MatMul_2MatMulinputssplit:output:2*
T0*(
_output_shapes
:??????????]
MatMul_3MatMulinputssplit:output:3*
T0*(
_output_shapes
:??????????S
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : s
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes	
:?*
dtype0?
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_spliti
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*(
_output_shapes
:??????????m
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*(
_output_shapes
:??????????m
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*(
_output_shapes
:??????????m
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*(
_output_shapes
:??????????h
ReadVariableOpReadVariableOpreadvariableop_resource* 
_output_shapes
:
??*
dtype0d
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_maskg
MatMul_4MatMulstates_0strided_slice:output:0*
T0*(
_output_shapes
:??????????e
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*(
_output_shapes
:??????????J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>L
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?V
MulMuladd:z:0Const:output:0*
T0*(
_output_shapes
:??????????\
Add_1AddV2Mul:z:0Const_1:output:0*
T0*(
_output_shapes
:??????????\
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:??????????T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*(
_output_shapes
:??????????j
ReadVariableOp_1ReadVariableOpreadvariableop_resource* 
_output_shapes
:
??*
dtype0f
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   h
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_maski
MatMul_5MatMulstates_0strided_slice_1:output:0*
T0*(
_output_shapes
:??????????i
add_2AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*(
_output_shapes
:??????????L
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>L
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?\
Mul_1Mul	add_2:z:0Const_2:output:0*
T0*(
_output_shapes
:??????????^
Add_3AddV2	Mul_1:z:0Const_3:output:0*
T0*(
_output_shapes
:??????????^
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:??????????V
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*(
_output_shapes
:??????????^
mul_2Mulclip_by_value_1:z:0states_1*
T0*(
_output_shapes
:??????????j
ReadVariableOp_2ReadVariableOpreadvariableop_resource* 
_output_shapes
:
??*
dtype0f
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  h
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_maski
MatMul_6MatMulstates_0strided_slice_2:output:0*
T0*(
_output_shapes
:??????????i
add_4AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*(
_output_shapes
:??????????J
TanhTanh	add_4:z:0*
T0*(
_output_shapes
:??????????\
mul_3Mulclip_by_value:z:0Tanh:y:0*
T0*(
_output_shapes
:??????????W
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*(
_output_shapes
:??????????j
ReadVariableOp_3ReadVariableOpreadvariableop_resource* 
_output_shapes
:
??*
dtype0f
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  h
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_maski
MatMul_7MatMulstates_0strided_slice_3:output:0*
T0*(
_output_shapes
:??????????i
add_6AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*(
_output_shapes
:??????????L
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>L
Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?\
Mul_4Mul	add_6:z:0Const_4:output:0*
T0*(
_output_shapes
:??????????^
Add_7AddV2	Mul_4:z:0Const_5:output:0*
T0*(
_output_shapes
:??????????^
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:??????????V
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*(
_output_shapes
:??????????L
Tanh_1Tanh	add_5:z:0*
T0*(
_output_shapes
:??????????`
mul_5Mulclip_by_value_2:z:0
Tanh_1:y:0*
T0*(
_output_shapes
:??????????Y
IdentityIdentity	mul_5:z:0^NoOp*
T0*(
_output_shapes
:??????????[

Identity_1Identity	mul_5:z:0^NoOp*
T0*(
_output_shapes
:??????????[

Identity_2Identity	add_5:z:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:??????????:??????????:??????????: : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/0:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/1
??
?
A__inference_lstm_2_layer_call_and_return_conditional_losses_57833
inputs_0<
)lstm_cell_2_split_readvariableop_resource:	?:
+lstm_cell_2_split_1_readvariableop_resource:	?7
#lstm_cell_2_readvariableop_resource:
??
identity??lstm_cell_2/ReadVariableOp?lstm_cell_2/ReadVariableOp_1?lstm_cell_2/ReadVariableOp_2?lstm_cell_2/ReadVariableOp_3? lstm_cell_2/split/ReadVariableOp?"lstm_cell_2/split_1/ReadVariableOp?while=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????S
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask]
lstm_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
 lstm_cell_2/split/ReadVariableOpReadVariableOp)lstm_cell_2_split_readvariableop_resource*
_output_shapes
:	?*
dtype0?
lstm_cell_2/splitSplit$lstm_cell_2/split/split_dim:output:0(lstm_cell_2/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	?:	?:	?:	?*
	num_split?
lstm_cell_2/MatMulMatMulstrided_slice_2:output:0lstm_cell_2/split:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_2/MatMul_1MatMulstrided_slice_2:output:0lstm_cell_2/split:output:1*
T0*(
_output_shapes
:???????????
lstm_cell_2/MatMul_2MatMulstrided_slice_2:output:0lstm_cell_2/split:output:2*
T0*(
_output_shapes
:???????????
lstm_cell_2/MatMul_3MatMulstrided_slice_2:output:0lstm_cell_2/split:output:3*
T0*(
_output_shapes
:??????????_
lstm_cell_2/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
"lstm_cell_2/split_1/ReadVariableOpReadVariableOp+lstm_cell_2_split_1_readvariableop_resource*
_output_shapes	
:?*
dtype0?
lstm_cell_2/split_1Split&lstm_cell_2/split_1/split_dim:output:0*lstm_cell_2/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split?
lstm_cell_2/BiasAddBiasAddlstm_cell_2/MatMul:product:0lstm_cell_2/split_1:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_2/BiasAdd_1BiasAddlstm_cell_2/MatMul_1:product:0lstm_cell_2/split_1:output:1*
T0*(
_output_shapes
:???????????
lstm_cell_2/BiasAdd_2BiasAddlstm_cell_2/MatMul_2:product:0lstm_cell_2/split_1:output:2*
T0*(
_output_shapes
:???????????
lstm_cell_2/BiasAdd_3BiasAddlstm_cell_2/MatMul_3:product:0lstm_cell_2/split_1:output:3*
T0*(
_output_shapes
:???????????
lstm_cell_2/ReadVariableOpReadVariableOp#lstm_cell_2_readvariableop_resource* 
_output_shapes
:
??*
dtype0p
lstm_cell_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        r
!lstm_cell_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       r
!lstm_cell_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
lstm_cell_2/strided_sliceStridedSlice"lstm_cell_2/ReadVariableOp:value:0(lstm_cell_2/strided_slice/stack:output:0*lstm_cell_2/strided_slice/stack_1:output:0*lstm_cell_2/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
lstm_cell_2/MatMul_4MatMulzeros:output:0"lstm_cell_2/strided_slice:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_2/addAddV2lstm_cell_2/BiasAdd:output:0lstm_cell_2/MatMul_4:product:0*
T0*(
_output_shapes
:??????????V
lstm_cell_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>X
lstm_cell_2/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?z
lstm_cell_2/MulMullstm_cell_2/add:z:0lstm_cell_2/Const:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_2/Add_1AddV2lstm_cell_2/Mul:z:0lstm_cell_2/Const_1:output:0*
T0*(
_output_shapes
:??????????h
#lstm_cell_2/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
!lstm_cell_2/clip_by_value/MinimumMinimumlstm_cell_2/Add_1:z:0,lstm_cell_2/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:??????????`
lstm_cell_2/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
lstm_cell_2/clip_by_valueMaximum%lstm_cell_2/clip_by_value/Minimum:z:0$lstm_cell_2/clip_by_value/y:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_2/ReadVariableOp_1ReadVariableOp#lstm_cell_2_readvariableop_resource* 
_output_shapes
:
??*
dtype0r
!lstm_cell_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       t
#lstm_cell_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       t
#lstm_cell_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
lstm_cell_2/strided_slice_1StridedSlice$lstm_cell_2/ReadVariableOp_1:value:0*lstm_cell_2/strided_slice_1/stack:output:0,lstm_cell_2/strided_slice_1/stack_1:output:0,lstm_cell_2/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
lstm_cell_2/MatMul_5MatMulzeros:output:0$lstm_cell_2/strided_slice_1:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_2/add_2AddV2lstm_cell_2/BiasAdd_1:output:0lstm_cell_2/MatMul_5:product:0*
T0*(
_output_shapes
:??????????X
lstm_cell_2/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>X
lstm_cell_2/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ??
lstm_cell_2/Mul_1Mullstm_cell_2/add_2:z:0lstm_cell_2/Const_2:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_2/Add_3AddV2lstm_cell_2/Mul_1:z:0lstm_cell_2/Const_3:output:0*
T0*(
_output_shapes
:??????????j
%lstm_cell_2/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
#lstm_cell_2/clip_by_value_1/MinimumMinimumlstm_cell_2/Add_3:z:0.lstm_cell_2/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:??????????b
lstm_cell_2/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
lstm_cell_2/clip_by_value_1Maximum'lstm_cell_2/clip_by_value_1/Minimum:z:0&lstm_cell_2/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:??????????~
lstm_cell_2/mul_2Mullstm_cell_2/clip_by_value_1:z:0zeros_1:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_2/ReadVariableOp_2ReadVariableOp#lstm_cell_2_readvariableop_resource* 
_output_shapes
:
??*
dtype0r
!lstm_cell_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       t
#lstm_cell_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       t
#lstm_cell_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
lstm_cell_2/strided_slice_2StridedSlice$lstm_cell_2/ReadVariableOp_2:value:0*lstm_cell_2/strided_slice_2/stack:output:0,lstm_cell_2/strided_slice_2/stack_1:output:0,lstm_cell_2/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
lstm_cell_2/MatMul_6MatMulzeros:output:0$lstm_cell_2/strided_slice_2:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_2/add_4AddV2lstm_cell_2/BiasAdd_2:output:0lstm_cell_2/MatMul_6:product:0*
T0*(
_output_shapes
:??????????b
lstm_cell_2/TanhTanhlstm_cell_2/add_4:z:0*
T0*(
_output_shapes
:???????????
lstm_cell_2/mul_3Mullstm_cell_2/clip_by_value:z:0lstm_cell_2/Tanh:y:0*
T0*(
_output_shapes
:??????????{
lstm_cell_2/add_5AddV2lstm_cell_2/mul_2:z:0lstm_cell_2/mul_3:z:0*
T0*(
_output_shapes
:???????????
lstm_cell_2/ReadVariableOp_3ReadVariableOp#lstm_cell_2_readvariableop_resource* 
_output_shapes
:
??*
dtype0r
!lstm_cell_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       t
#lstm_cell_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        t
#lstm_cell_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
lstm_cell_2/strided_slice_3StridedSlice$lstm_cell_2/ReadVariableOp_3:value:0*lstm_cell_2/strided_slice_3/stack:output:0,lstm_cell_2/strided_slice_3/stack_1:output:0,lstm_cell_2/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
lstm_cell_2/MatMul_7MatMulzeros:output:0$lstm_cell_2/strided_slice_3:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_2/add_6AddV2lstm_cell_2/BiasAdd_3:output:0lstm_cell_2/MatMul_7:product:0*
T0*(
_output_shapes
:??????????X
lstm_cell_2/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>X
lstm_cell_2/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ??
lstm_cell_2/Mul_4Mullstm_cell_2/add_6:z:0lstm_cell_2/Const_4:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_2/Add_7AddV2lstm_cell_2/Mul_4:z:0lstm_cell_2/Const_5:output:0*
T0*(
_output_shapes
:??????????j
%lstm_cell_2/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
#lstm_cell_2/clip_by_value_2/MinimumMinimumlstm_cell_2/Add_7:z:0.lstm_cell_2/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:??????????b
lstm_cell_2/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
lstm_cell_2/clip_by_value_2Maximum'lstm_cell_2/clip_by_value_2/Minimum:z:0&lstm_cell_2/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:??????????d
lstm_cell_2/Tanh_1Tanhlstm_cell_2/add_5:z:0*
T0*(
_output_shapes
:???????????
lstm_cell_2/mul_5Mullstm_cell_2/clip_by_value_2:z:0lstm_cell_2/Tanh_1:y:0*
T0*(
_output_shapes
:??????????n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)lstm_cell_2_split_readvariableop_resource+lstm_cell_2_split_1_readvariableop_resource#lstm_cell_2_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_57693*
condR
while_cond_57692*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:???????????????????*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:???????????????????l
IdentityIdentitytranspose_1:y:0^NoOp*
T0*5
_output_shapes#
!:????????????????????
NoOpNoOp^lstm_cell_2/ReadVariableOp^lstm_cell_2/ReadVariableOp_1^lstm_cell_2/ReadVariableOp_2^lstm_cell_2/ReadVariableOp_3!^lstm_cell_2/split/ReadVariableOp#^lstm_cell_2/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????: : : 28
lstm_cell_2/ReadVariableOplstm_cell_2/ReadVariableOp2<
lstm_cell_2/ReadVariableOp_1lstm_cell_2/ReadVariableOp_12<
lstm_cell_2/ReadVariableOp_2lstm_cell_2/ReadVariableOp_22<
lstm_cell_2/ReadVariableOp_3lstm_cell_2/ReadVariableOp_32D
 lstm_cell_2/split/ReadVariableOp lstm_cell_2/split/ReadVariableOp2H
"lstm_cell_2/split_1/ReadVariableOp"lstm_cell_2/split_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :??????????????????
"
_user_specified_name
inputs/0
?

?
#__inference_signature_wrapper_56227
lstm_2_input
unknown:	?
	unknown_0:	?
	unknown_1:
??
	unknown_2:
??
	unknown_3:	?
	unknown_4:
??
	unknown_5:
??
	unknown_6:	?
	unknown_7:	?
	unknown_8:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalllstm_2_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *)
f$R"
 __inference__wrapped_model_53944o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
+
_output_shapes
:?????????
&
_user_specified_namelstm_2_input
??
?
A__inference_lstm_3_layer_call_and_return_conditional_losses_59413

inputs=
)lstm_cell_3_split_readvariableop_resource:
??:
+lstm_cell_3_split_1_readvariableop_resource:	?7
#lstm_cell_3_readvariableop_resource:
??
identity??lstm_cell_3/ReadVariableOp?lstm_cell_3/ReadVariableOp_1?lstm_cell_3/ReadVariableOp_2?lstm_cell_3/ReadVariableOp_3? lstm_cell_3/split/ReadVariableOp?"lstm_cell_3/split_1/ReadVariableOp?while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????S
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:??????????D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask]
lstm_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
 lstm_cell_3/split/ReadVariableOpReadVariableOp)lstm_cell_3_split_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
lstm_cell_3/splitSplit$lstm_cell_3/split/split_dim:output:0(lstm_cell_3/split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
??:
??:
??:
??*
	num_split?
lstm_cell_3/MatMulMatMulstrided_slice_2:output:0lstm_cell_3/split:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_3/MatMul_1MatMulstrided_slice_2:output:0lstm_cell_3/split:output:1*
T0*(
_output_shapes
:???????????
lstm_cell_3/MatMul_2MatMulstrided_slice_2:output:0lstm_cell_3/split:output:2*
T0*(
_output_shapes
:???????????
lstm_cell_3/MatMul_3MatMulstrided_slice_2:output:0lstm_cell_3/split:output:3*
T0*(
_output_shapes
:??????????_
lstm_cell_3/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
"lstm_cell_3/split_1/ReadVariableOpReadVariableOp+lstm_cell_3_split_1_readvariableop_resource*
_output_shapes	
:?*
dtype0?
lstm_cell_3/split_1Split&lstm_cell_3/split_1/split_dim:output:0*lstm_cell_3/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split?
lstm_cell_3/BiasAddBiasAddlstm_cell_3/MatMul:product:0lstm_cell_3/split_1:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_3/BiasAdd_1BiasAddlstm_cell_3/MatMul_1:product:0lstm_cell_3/split_1:output:1*
T0*(
_output_shapes
:???????????
lstm_cell_3/BiasAdd_2BiasAddlstm_cell_3/MatMul_2:product:0lstm_cell_3/split_1:output:2*
T0*(
_output_shapes
:???????????
lstm_cell_3/BiasAdd_3BiasAddlstm_cell_3/MatMul_3:product:0lstm_cell_3/split_1:output:3*
T0*(
_output_shapes
:???????????
lstm_cell_3/ReadVariableOpReadVariableOp#lstm_cell_3_readvariableop_resource* 
_output_shapes
:
??*
dtype0p
lstm_cell_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        r
!lstm_cell_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   r
!lstm_cell_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
lstm_cell_3/strided_sliceStridedSlice"lstm_cell_3/ReadVariableOp:value:0(lstm_cell_3/strided_slice/stack:output:0*lstm_cell_3/strided_slice/stack_1:output:0*lstm_cell_3/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
lstm_cell_3/MatMul_4MatMulzeros:output:0"lstm_cell_3/strided_slice:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_3/addAddV2lstm_cell_3/BiasAdd:output:0lstm_cell_3/MatMul_4:product:0*
T0*(
_output_shapes
:??????????V
lstm_cell_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>X
lstm_cell_3/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?z
lstm_cell_3/MulMullstm_cell_3/add:z:0lstm_cell_3/Const:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_3/Add_1AddV2lstm_cell_3/Mul:z:0lstm_cell_3/Const_1:output:0*
T0*(
_output_shapes
:??????????h
#lstm_cell_3/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
!lstm_cell_3/clip_by_value/MinimumMinimumlstm_cell_3/Add_1:z:0,lstm_cell_3/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:??????????`
lstm_cell_3/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
lstm_cell_3/clip_by_valueMaximum%lstm_cell_3/clip_by_value/Minimum:z:0$lstm_cell_3/clip_by_value/y:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_3/ReadVariableOp_1ReadVariableOp#lstm_cell_3_readvariableop_resource* 
_output_shapes
:
??*
dtype0r
!lstm_cell_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   t
#lstm_cell_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       t
#lstm_cell_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
lstm_cell_3/strided_slice_1StridedSlice$lstm_cell_3/ReadVariableOp_1:value:0*lstm_cell_3/strided_slice_1/stack:output:0,lstm_cell_3/strided_slice_1/stack_1:output:0,lstm_cell_3/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
lstm_cell_3/MatMul_5MatMulzeros:output:0$lstm_cell_3/strided_slice_1:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_3/add_2AddV2lstm_cell_3/BiasAdd_1:output:0lstm_cell_3/MatMul_5:product:0*
T0*(
_output_shapes
:??????????X
lstm_cell_3/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>X
lstm_cell_3/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ??
lstm_cell_3/Mul_1Mullstm_cell_3/add_2:z:0lstm_cell_3/Const_2:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_3/Add_3AddV2lstm_cell_3/Mul_1:z:0lstm_cell_3/Const_3:output:0*
T0*(
_output_shapes
:??????????j
%lstm_cell_3/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
#lstm_cell_3/clip_by_value_1/MinimumMinimumlstm_cell_3/Add_3:z:0.lstm_cell_3/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:??????????b
lstm_cell_3/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
lstm_cell_3/clip_by_value_1Maximum'lstm_cell_3/clip_by_value_1/Minimum:z:0&lstm_cell_3/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:??????????~
lstm_cell_3/mul_2Mullstm_cell_3/clip_by_value_1:z:0zeros_1:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_3/ReadVariableOp_2ReadVariableOp#lstm_cell_3_readvariableop_resource* 
_output_shapes
:
??*
dtype0r
!lstm_cell_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       t
#lstm_cell_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  t
#lstm_cell_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
lstm_cell_3/strided_slice_2StridedSlice$lstm_cell_3/ReadVariableOp_2:value:0*lstm_cell_3/strided_slice_2/stack:output:0,lstm_cell_3/strided_slice_2/stack_1:output:0,lstm_cell_3/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
lstm_cell_3/MatMul_6MatMulzeros:output:0$lstm_cell_3/strided_slice_2:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_3/add_4AddV2lstm_cell_3/BiasAdd_2:output:0lstm_cell_3/MatMul_6:product:0*
T0*(
_output_shapes
:??????????b
lstm_cell_3/TanhTanhlstm_cell_3/add_4:z:0*
T0*(
_output_shapes
:???????????
lstm_cell_3/mul_3Mullstm_cell_3/clip_by_value:z:0lstm_cell_3/Tanh:y:0*
T0*(
_output_shapes
:??????????{
lstm_cell_3/add_5AddV2lstm_cell_3/mul_2:z:0lstm_cell_3/mul_3:z:0*
T0*(
_output_shapes
:???????????
lstm_cell_3/ReadVariableOp_3ReadVariableOp#lstm_cell_3_readvariableop_resource* 
_output_shapes
:
??*
dtype0r
!lstm_cell_3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  t
#lstm_cell_3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        t
#lstm_cell_3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
lstm_cell_3/strided_slice_3StridedSlice$lstm_cell_3/ReadVariableOp_3:value:0*lstm_cell_3/strided_slice_3/stack:output:0,lstm_cell_3/strided_slice_3/stack_1:output:0,lstm_cell_3/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
lstm_cell_3/MatMul_7MatMulzeros:output:0$lstm_cell_3/strided_slice_3:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_3/add_6AddV2lstm_cell_3/BiasAdd_3:output:0lstm_cell_3/MatMul_7:product:0*
T0*(
_output_shapes
:??????????X
lstm_cell_3/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>X
lstm_cell_3/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ??
lstm_cell_3/Mul_4Mullstm_cell_3/add_6:z:0lstm_cell_3/Const_4:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_3/Add_7AddV2lstm_cell_3/Mul_4:z:0lstm_cell_3/Const_5:output:0*
T0*(
_output_shapes
:??????????j
%lstm_cell_3/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
#lstm_cell_3/clip_by_value_2/MinimumMinimumlstm_cell_3/Add_7:z:0.lstm_cell_3/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:??????????b
lstm_cell_3/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
lstm_cell_3/clip_by_value_2Maximum'lstm_cell_3/clip_by_value_2/Minimum:z:0&lstm_cell_3/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:??????????d
lstm_cell_3/Tanh_1Tanhlstm_cell_3/add_5:z:0*
T0*(
_output_shapes
:???????????
lstm_cell_3/mul_5Mullstm_cell_3/clip_by_value_2:z:0lstm_cell_3/Tanh_1:y:0*
T0*(
_output_shapes
:??????????n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   ?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)lstm_cell_3_split_readvariableop_resource+lstm_cell_3_split_1_readvariableop_resource#lstm_cell_3_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_59273*
condR
while_cond_59272*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????h
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp^lstm_cell_3/ReadVariableOp^lstm_cell_3/ReadVariableOp_1^lstm_cell_3/ReadVariableOp_2^lstm_cell_3/ReadVariableOp_3!^lstm_cell_3/split/ReadVariableOp#^lstm_cell_3/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:??????????: : : 28
lstm_cell_3/ReadVariableOplstm_cell_3/ReadVariableOp2<
lstm_cell_3/ReadVariableOp_1lstm_cell_3/ReadVariableOp_12<
lstm_cell_3/ReadVariableOp_2lstm_cell_3/ReadVariableOp_22<
lstm_cell_3/ReadVariableOp_3lstm_cell_3/ReadVariableOp_32D
 lstm_cell_3/split/ReadVariableOp lstm_cell_3/split/ReadVariableOp2H
"lstm_cell_3/split_1/ReadVariableOp"lstm_cell_3/split_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?J
?
F__inference_lstm_cell_2_layer_call_and_return_conditional_losses_54270

inputs

states
states_10
split_readvariableop_resource:	?.
split_1_readvariableop_resource:	?+
readvariableop_resource:
??
identity

identity_1

identity_2??ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?ReadVariableOp_3?split/ReadVariableOp?split_1/ReadVariableOpQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :s
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes
:	?*
dtype0?
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	?:	?:	?:	?*
	num_split[
MatMulMatMulinputssplit:output:0*
T0*(
_output_shapes
:??????????]
MatMul_1MatMulinputssplit:output:1*
T0*(
_output_shapes
:??????????]
MatMul_2MatMulinputssplit:output:2*
T0*(
_output_shapes
:??????????]
MatMul_3MatMulinputssplit:output:3*
T0*(
_output_shapes
:??????????S
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : s
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes	
:?*
dtype0?
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_spliti
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*(
_output_shapes
:??????????m
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*(
_output_shapes
:??????????m
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*(
_output_shapes
:??????????m
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*(
_output_shapes
:??????????h
ReadVariableOpReadVariableOpreadvariableop_resource* 
_output_shapes
:
??*
dtype0d
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_maske
MatMul_4MatMulstatesstrided_slice:output:0*
T0*(
_output_shapes
:??????????e
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*(
_output_shapes
:??????????J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>L
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?V
MulMuladd:z:0Const:output:0*
T0*(
_output_shapes
:??????????\
Add_1AddV2Mul:z:0Const_1:output:0*
T0*(
_output_shapes
:??????????\
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:??????????T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*(
_output_shapes
:??????????j
ReadVariableOp_1ReadVariableOpreadvariableop_resource* 
_output_shapes
:
??*
dtype0f
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_maskg
MatMul_5MatMulstatesstrided_slice_1:output:0*
T0*(
_output_shapes
:??????????i
add_2AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*(
_output_shapes
:??????????L
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>L
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?\
Mul_1Mul	add_2:z:0Const_2:output:0*
T0*(
_output_shapes
:??????????^
Add_3AddV2	Mul_1:z:0Const_3:output:0*
T0*(
_output_shapes
:??????????^
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:??????????V
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*(
_output_shapes
:??????????^
mul_2Mulclip_by_value_1:z:0states_1*
T0*(
_output_shapes
:??????????j
ReadVariableOp_2ReadVariableOpreadvariableop_resource* 
_output_shapes
:
??*
dtype0f
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_maskg
MatMul_6MatMulstatesstrided_slice_2:output:0*
T0*(
_output_shapes
:??????????i
add_4AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*(
_output_shapes
:??????????J
TanhTanh	add_4:z:0*
T0*(
_output_shapes
:??????????\
mul_3Mulclip_by_value:z:0Tanh:y:0*
T0*(
_output_shapes
:??????????W
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*(
_output_shapes
:??????????j
ReadVariableOp_3ReadVariableOpreadvariableop_resource* 
_output_shapes
:
??*
dtype0f
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_maskg
MatMul_7MatMulstatesstrided_slice_3:output:0*
T0*(
_output_shapes
:??????????i
add_6AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*(
_output_shapes
:??????????L
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>L
Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?\
Mul_4Mul	add_6:z:0Const_4:output:0*
T0*(
_output_shapes
:??????????^
Add_7AddV2	Mul_4:z:0Const_5:output:0*
T0*(
_output_shapes
:??????????^
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:??????????V
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*(
_output_shapes
:??????????L
Tanh_1Tanh	add_5:z:0*
T0*(
_output_shapes
:??????????`
mul_5Mulclip_by_value_2:z:0
Tanh_1:y:0*
T0*(
_output_shapes
:??????????Y
IdentityIdentity	mul_5:z:0^NoOp*
T0*(
_output_shapes
:??????????[

Identity_1Identity	mul_5:z:0^NoOp*
T0*(
_output_shapes
:??????????[

Identity_2Identity	add_5:z:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:?????????:??????????:??????????: : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:PL
(
_output_shapes
:??????????
 
_user_specified_namestates:PL
(
_output_shapes
:??????????
 
_user_specified_namestates
??
?
A__inference_lstm_3_layer_call_and_return_conditional_losses_59157

inputs=
)lstm_cell_3_split_readvariableop_resource:
??:
+lstm_cell_3_split_1_readvariableop_resource:	?7
#lstm_cell_3_readvariableop_resource:
??
identity??lstm_cell_3/ReadVariableOp?lstm_cell_3/ReadVariableOp_1?lstm_cell_3/ReadVariableOp_2?lstm_cell_3/ReadVariableOp_3? lstm_cell_3/split/ReadVariableOp?"lstm_cell_3/split_1/ReadVariableOp?while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????S
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:??????????D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask]
lstm_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
 lstm_cell_3/split/ReadVariableOpReadVariableOp)lstm_cell_3_split_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
lstm_cell_3/splitSplit$lstm_cell_3/split/split_dim:output:0(lstm_cell_3/split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
??:
??:
??:
??*
	num_split?
lstm_cell_3/MatMulMatMulstrided_slice_2:output:0lstm_cell_3/split:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_3/MatMul_1MatMulstrided_slice_2:output:0lstm_cell_3/split:output:1*
T0*(
_output_shapes
:???????????
lstm_cell_3/MatMul_2MatMulstrided_slice_2:output:0lstm_cell_3/split:output:2*
T0*(
_output_shapes
:???????????
lstm_cell_3/MatMul_3MatMulstrided_slice_2:output:0lstm_cell_3/split:output:3*
T0*(
_output_shapes
:??????????_
lstm_cell_3/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
"lstm_cell_3/split_1/ReadVariableOpReadVariableOp+lstm_cell_3_split_1_readvariableop_resource*
_output_shapes	
:?*
dtype0?
lstm_cell_3/split_1Split&lstm_cell_3/split_1/split_dim:output:0*lstm_cell_3/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split?
lstm_cell_3/BiasAddBiasAddlstm_cell_3/MatMul:product:0lstm_cell_3/split_1:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_3/BiasAdd_1BiasAddlstm_cell_3/MatMul_1:product:0lstm_cell_3/split_1:output:1*
T0*(
_output_shapes
:???????????
lstm_cell_3/BiasAdd_2BiasAddlstm_cell_3/MatMul_2:product:0lstm_cell_3/split_1:output:2*
T0*(
_output_shapes
:???????????
lstm_cell_3/BiasAdd_3BiasAddlstm_cell_3/MatMul_3:product:0lstm_cell_3/split_1:output:3*
T0*(
_output_shapes
:???????????
lstm_cell_3/ReadVariableOpReadVariableOp#lstm_cell_3_readvariableop_resource* 
_output_shapes
:
??*
dtype0p
lstm_cell_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        r
!lstm_cell_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   r
!lstm_cell_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
lstm_cell_3/strided_sliceStridedSlice"lstm_cell_3/ReadVariableOp:value:0(lstm_cell_3/strided_slice/stack:output:0*lstm_cell_3/strided_slice/stack_1:output:0*lstm_cell_3/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
lstm_cell_3/MatMul_4MatMulzeros:output:0"lstm_cell_3/strided_slice:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_3/addAddV2lstm_cell_3/BiasAdd:output:0lstm_cell_3/MatMul_4:product:0*
T0*(
_output_shapes
:??????????V
lstm_cell_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>X
lstm_cell_3/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?z
lstm_cell_3/MulMullstm_cell_3/add:z:0lstm_cell_3/Const:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_3/Add_1AddV2lstm_cell_3/Mul:z:0lstm_cell_3/Const_1:output:0*
T0*(
_output_shapes
:??????????h
#lstm_cell_3/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
!lstm_cell_3/clip_by_value/MinimumMinimumlstm_cell_3/Add_1:z:0,lstm_cell_3/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:??????????`
lstm_cell_3/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
lstm_cell_3/clip_by_valueMaximum%lstm_cell_3/clip_by_value/Minimum:z:0$lstm_cell_3/clip_by_value/y:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_3/ReadVariableOp_1ReadVariableOp#lstm_cell_3_readvariableop_resource* 
_output_shapes
:
??*
dtype0r
!lstm_cell_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   t
#lstm_cell_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       t
#lstm_cell_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
lstm_cell_3/strided_slice_1StridedSlice$lstm_cell_3/ReadVariableOp_1:value:0*lstm_cell_3/strided_slice_1/stack:output:0,lstm_cell_3/strided_slice_1/stack_1:output:0,lstm_cell_3/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
lstm_cell_3/MatMul_5MatMulzeros:output:0$lstm_cell_3/strided_slice_1:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_3/add_2AddV2lstm_cell_3/BiasAdd_1:output:0lstm_cell_3/MatMul_5:product:0*
T0*(
_output_shapes
:??????????X
lstm_cell_3/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>X
lstm_cell_3/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ??
lstm_cell_3/Mul_1Mullstm_cell_3/add_2:z:0lstm_cell_3/Const_2:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_3/Add_3AddV2lstm_cell_3/Mul_1:z:0lstm_cell_3/Const_3:output:0*
T0*(
_output_shapes
:??????????j
%lstm_cell_3/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
#lstm_cell_3/clip_by_value_1/MinimumMinimumlstm_cell_3/Add_3:z:0.lstm_cell_3/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:??????????b
lstm_cell_3/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
lstm_cell_3/clip_by_value_1Maximum'lstm_cell_3/clip_by_value_1/Minimum:z:0&lstm_cell_3/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:??????????~
lstm_cell_3/mul_2Mullstm_cell_3/clip_by_value_1:z:0zeros_1:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_3/ReadVariableOp_2ReadVariableOp#lstm_cell_3_readvariableop_resource* 
_output_shapes
:
??*
dtype0r
!lstm_cell_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       t
#lstm_cell_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  t
#lstm_cell_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
lstm_cell_3/strided_slice_2StridedSlice$lstm_cell_3/ReadVariableOp_2:value:0*lstm_cell_3/strided_slice_2/stack:output:0,lstm_cell_3/strided_slice_2/stack_1:output:0,lstm_cell_3/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
lstm_cell_3/MatMul_6MatMulzeros:output:0$lstm_cell_3/strided_slice_2:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_3/add_4AddV2lstm_cell_3/BiasAdd_2:output:0lstm_cell_3/MatMul_6:product:0*
T0*(
_output_shapes
:??????????b
lstm_cell_3/TanhTanhlstm_cell_3/add_4:z:0*
T0*(
_output_shapes
:???????????
lstm_cell_3/mul_3Mullstm_cell_3/clip_by_value:z:0lstm_cell_3/Tanh:y:0*
T0*(
_output_shapes
:??????????{
lstm_cell_3/add_5AddV2lstm_cell_3/mul_2:z:0lstm_cell_3/mul_3:z:0*
T0*(
_output_shapes
:???????????
lstm_cell_3/ReadVariableOp_3ReadVariableOp#lstm_cell_3_readvariableop_resource* 
_output_shapes
:
??*
dtype0r
!lstm_cell_3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  t
#lstm_cell_3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        t
#lstm_cell_3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
lstm_cell_3/strided_slice_3StridedSlice$lstm_cell_3/ReadVariableOp_3:value:0*lstm_cell_3/strided_slice_3/stack:output:0,lstm_cell_3/strided_slice_3/stack_1:output:0,lstm_cell_3/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
lstm_cell_3/MatMul_7MatMulzeros:output:0$lstm_cell_3/strided_slice_3:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_3/add_6AddV2lstm_cell_3/BiasAdd_3:output:0lstm_cell_3/MatMul_7:product:0*
T0*(
_output_shapes
:??????????X
lstm_cell_3/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>X
lstm_cell_3/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ??
lstm_cell_3/Mul_4Mullstm_cell_3/add_6:z:0lstm_cell_3/Const_4:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_3/Add_7AddV2lstm_cell_3/Mul_4:z:0lstm_cell_3/Const_5:output:0*
T0*(
_output_shapes
:??????????j
%lstm_cell_3/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
#lstm_cell_3/clip_by_value_2/MinimumMinimumlstm_cell_3/Add_7:z:0.lstm_cell_3/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:??????????b
lstm_cell_3/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
lstm_cell_3/clip_by_value_2Maximum'lstm_cell_3/clip_by_value_2/Minimum:z:0&lstm_cell_3/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:??????????d
lstm_cell_3/Tanh_1Tanhlstm_cell_3/add_5:z:0*
T0*(
_output_shapes
:???????????
lstm_cell_3/mul_5Mullstm_cell_3/clip_by_value_2:z:0lstm_cell_3/Tanh_1:y:0*
T0*(
_output_shapes
:??????????n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   ?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)lstm_cell_3_split_readvariableop_resource+lstm_cell_3_split_1_readvariableop_resource#lstm_cell_3_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_59017*
condR
while_cond_59016*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????h
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp^lstm_cell_3/ReadVariableOp^lstm_cell_3/ReadVariableOp_1^lstm_cell_3/ReadVariableOp_2^lstm_cell_3/ReadVariableOp_3!^lstm_cell_3/split/ReadVariableOp#^lstm_cell_3/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:??????????: : : 28
lstm_cell_3/ReadVariableOplstm_cell_3/ReadVariableOp2<
lstm_cell_3/ReadVariableOp_1lstm_cell_3/ReadVariableOp_12<
lstm_cell_3/ReadVariableOp_2lstm_cell_3/ReadVariableOp_22<
lstm_cell_3/ReadVariableOp_3lstm_cell_3/ReadVariableOp_32D
 lstm_cell_3/split/ReadVariableOp lstm_cell_3/split/ReadVariableOp2H
"lstm_cell_3/split_1/ReadVariableOp"lstm_cell_3/split_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
+__inference_lstm_cell_3_layer_call_fn_59904

inputs
states_0
states_1
unknown:
??
	unknown_0:	?
	unknown_1:
??
identity

identity_1

identity_2??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:??????????:??????????:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_lstm_cell_3_layer_call_and_return_conditional_losses_54530p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????r

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:??????????r

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:??????????:??????????:??????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/0:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/1
?
?
&__inference_lstm_3_layer_call_fn_59457

inputs
unknown:
??
	unknown_0:	?
	unknown_1:
??
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_lstm_3_layer_call_and_return_conditional_losses_55748p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:??????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
while_cond_59016
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_59016___redundant_placeholder03
/while_while_cond_59016___redundant_placeholder13
/while_while_cond_59016___redundant_placeholder23
/while_while_cond_59016___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?"
?
while_body_54544
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0-
while_lstm_cell_3_54568_0:
??(
while_lstm_cell_3_54570_0:	?-
while_lstm_cell_3_54572_0:
??
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor+
while_lstm_cell_3_54568:
??&
while_lstm_cell_3_54570:	?+
while_lstm_cell_3_54572:
????)while/lstm_cell_3/StatefulPartitionedCall?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype0?
)while/lstm_cell_3/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_3_54568_0while_lstm_cell_3_54570_0while_lstm_cell_3_54572_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:??????????:??????????:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_lstm_cell_3_layer_call_and_return_conditional_losses_54530?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder2while/lstm_cell_3/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:???M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_4Identity2while/lstm_cell_3/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:???????????
while/Identity_5Identity2while/lstm_cell_3/StatefulPartitionedCall:output:2^while/NoOp*
T0*(
_output_shapes
:??????????x

while/NoOpNoOp*^while/lstm_cell_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"4
while_lstm_cell_3_54568while_lstm_cell_3_54568_0"4
while_lstm_cell_3_54570while_lstm_cell_3_54570_0"4
while_lstm_cell_3_54572while_lstm_cell_3_54572_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2V
)while/lstm_cell_3/StatefulPartitionedCall)while/lstm_cell_3/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?
?
G__inference_sequential_1_layer_call_and_return_conditional_losses_55436

inputs
lstm_2_55131:	?
lstm_2_55133:	? 
lstm_2_55135:
?? 
lstm_3_55394:
??
lstm_3_55396:	? 
lstm_3_55398:
??!
dense_2_55413:
??
dense_2_55415:	? 
dense_3_55430:	?
dense_3_55432:
identity??dense_2/StatefulPartitionedCall?dense_3/StatefulPartitionedCall?lstm_2/StatefulPartitionedCall?lstm_3/StatefulPartitionedCall?
lstm_2/StatefulPartitionedCallStatefulPartitionedCallinputslstm_2_55131lstm_2_55133lstm_2_55135*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_lstm_2_layer_call_and_return_conditional_losses_55130?
lstm_3/StatefulPartitionedCallStatefulPartitionedCall'lstm_2/StatefulPartitionedCall:output:0lstm_3_55394lstm_3_55396lstm_3_55398*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_lstm_3_layer_call_and_return_conditional_losses_55393?
dense_2/StatefulPartitionedCallStatefulPartitionedCall'lstm_3/StatefulPartitionedCall:output:0dense_2_55413dense_2_55415*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_55412?
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_55430dense_3_55432*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_55429w
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall^lstm_2/StatefulPartitionedCall^lstm_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????: : : : : : : : : : 2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2@
lstm_2/StatefulPartitionedCalllstm_2/StatefulPartitionedCall2@
lstm_3/StatefulPartitionedCalllstm_3/StatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?{
?	
while_body_57949
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_lstm_cell_2_split_readvariableop_resource_0:	?B
3while_lstm_cell_2_split_1_readvariableop_resource_0:	??
+while_lstm_cell_2_readvariableop_resource_0:
??
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_lstm_cell_2_split_readvariableop_resource:	?@
1while_lstm_cell_2_split_1_readvariableop_resource:	?=
)while_lstm_cell_2_readvariableop_resource:
???? while/lstm_cell_2/ReadVariableOp?"while/lstm_cell_2/ReadVariableOp_1?"while/lstm_cell_2/ReadVariableOp_2?"while/lstm_cell_2/ReadVariableOp_3?&while/lstm_cell_2/split/ReadVariableOp?(while/lstm_cell_2/split_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0c
!while/lstm_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
&while/lstm_cell_2/split/ReadVariableOpReadVariableOp1while_lstm_cell_2_split_readvariableop_resource_0*
_output_shapes
:	?*
dtype0?
while/lstm_cell_2/splitSplit*while/lstm_cell_2/split/split_dim:output:0.while/lstm_cell_2/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	?:	?:	?:	?*
	num_split?
while/lstm_cell_2/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_2/split:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_2/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_2/split:output:1*
T0*(
_output_shapes
:???????????
while/lstm_cell_2/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_2/split:output:2*
T0*(
_output_shapes
:???????????
while/lstm_cell_2/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_2/split:output:3*
T0*(
_output_shapes
:??????????e
#while/lstm_cell_2/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
(while/lstm_cell_2/split_1/ReadVariableOpReadVariableOp3while_lstm_cell_2_split_1_readvariableop_resource_0*
_output_shapes	
:?*
dtype0?
while/lstm_cell_2/split_1Split,while/lstm_cell_2/split_1/split_dim:output:00while/lstm_cell_2/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split?
while/lstm_cell_2/BiasAddBiasAdd"while/lstm_cell_2/MatMul:product:0"while/lstm_cell_2/split_1:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_2/BiasAdd_1BiasAdd$while/lstm_cell_2/MatMul_1:product:0"while/lstm_cell_2/split_1:output:1*
T0*(
_output_shapes
:???????????
while/lstm_cell_2/BiasAdd_2BiasAdd$while/lstm_cell_2/MatMul_2:product:0"while/lstm_cell_2/split_1:output:2*
T0*(
_output_shapes
:???????????
while/lstm_cell_2/BiasAdd_3BiasAdd$while/lstm_cell_2/MatMul_3:product:0"while/lstm_cell_2/split_1:output:3*
T0*(
_output_shapes
:???????????
 while/lstm_cell_2/ReadVariableOpReadVariableOp+while_lstm_cell_2_readvariableop_resource_0* 
_output_shapes
:
??*
dtype0v
%while/lstm_cell_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        x
'while/lstm_cell_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       x
'while/lstm_cell_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
while/lstm_cell_2/strided_sliceStridedSlice(while/lstm_cell_2/ReadVariableOp:value:0.while/lstm_cell_2/strided_slice/stack:output:00while/lstm_cell_2/strided_slice/stack_1:output:00while/lstm_cell_2/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
while/lstm_cell_2/MatMul_4MatMulwhile_placeholder_2(while/lstm_cell_2/strided_slice:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_2/addAddV2"while/lstm_cell_2/BiasAdd:output:0$while/lstm_cell_2/MatMul_4:product:0*
T0*(
_output_shapes
:??????????\
while/lstm_cell_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>^
while/lstm_cell_2/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ??
while/lstm_cell_2/MulMulwhile/lstm_cell_2/add:z:0 while/lstm_cell_2/Const:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_2/Add_1AddV2while/lstm_cell_2/Mul:z:0"while/lstm_cell_2/Const_1:output:0*
T0*(
_output_shapes
:??????????n
)while/lstm_cell_2/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
'while/lstm_cell_2/clip_by_value/MinimumMinimumwhile/lstm_cell_2/Add_1:z:02while/lstm_cell_2/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:??????????f
!while/lstm_cell_2/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
while/lstm_cell_2/clip_by_valueMaximum+while/lstm_cell_2/clip_by_value/Minimum:z:0*while/lstm_cell_2/clip_by_value/y:output:0*
T0*(
_output_shapes
:???????????
"while/lstm_cell_2/ReadVariableOp_1ReadVariableOp+while_lstm_cell_2_readvariableop_resource_0* 
_output_shapes
:
??*
dtype0x
'while/lstm_cell_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       z
)while/lstm_cell_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       z
)while/lstm_cell_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
!while/lstm_cell_2/strided_slice_1StridedSlice*while/lstm_cell_2/ReadVariableOp_1:value:00while/lstm_cell_2/strided_slice_1/stack:output:02while/lstm_cell_2/strided_slice_1/stack_1:output:02while/lstm_cell_2/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
while/lstm_cell_2/MatMul_5MatMulwhile_placeholder_2*while/lstm_cell_2/strided_slice_1:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_2/add_2AddV2$while/lstm_cell_2/BiasAdd_1:output:0$while/lstm_cell_2/MatMul_5:product:0*
T0*(
_output_shapes
:??????????^
while/lstm_cell_2/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>^
while/lstm_cell_2/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ??
while/lstm_cell_2/Mul_1Mulwhile/lstm_cell_2/add_2:z:0"while/lstm_cell_2/Const_2:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_2/Add_3AddV2while/lstm_cell_2/Mul_1:z:0"while/lstm_cell_2/Const_3:output:0*
T0*(
_output_shapes
:??????????p
+while/lstm_cell_2/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
)while/lstm_cell_2/clip_by_value_1/MinimumMinimumwhile/lstm_cell_2/Add_3:z:04while/lstm_cell_2/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:??????????h
#while/lstm_cell_2/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
!while/lstm_cell_2/clip_by_value_1Maximum-while/lstm_cell_2/clip_by_value_1/Minimum:z:0,while/lstm_cell_2/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_2/mul_2Mul%while/lstm_cell_2/clip_by_value_1:z:0while_placeholder_3*
T0*(
_output_shapes
:???????????
"while/lstm_cell_2/ReadVariableOp_2ReadVariableOp+while_lstm_cell_2_readvariableop_resource_0* 
_output_shapes
:
??*
dtype0x
'while/lstm_cell_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       z
)while/lstm_cell_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       z
)while/lstm_cell_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
!while/lstm_cell_2/strided_slice_2StridedSlice*while/lstm_cell_2/ReadVariableOp_2:value:00while/lstm_cell_2/strided_slice_2/stack:output:02while/lstm_cell_2/strided_slice_2/stack_1:output:02while/lstm_cell_2/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
while/lstm_cell_2/MatMul_6MatMulwhile_placeholder_2*while/lstm_cell_2/strided_slice_2:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_2/add_4AddV2$while/lstm_cell_2/BiasAdd_2:output:0$while/lstm_cell_2/MatMul_6:product:0*
T0*(
_output_shapes
:??????????n
while/lstm_cell_2/TanhTanhwhile/lstm_cell_2/add_4:z:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_2/mul_3Mul#while/lstm_cell_2/clip_by_value:z:0while/lstm_cell_2/Tanh:y:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_2/add_5AddV2while/lstm_cell_2/mul_2:z:0while/lstm_cell_2/mul_3:z:0*
T0*(
_output_shapes
:???????????
"while/lstm_cell_2/ReadVariableOp_3ReadVariableOp+while_lstm_cell_2_readvariableop_resource_0* 
_output_shapes
:
??*
dtype0x
'while/lstm_cell_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       z
)while/lstm_cell_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        z
)while/lstm_cell_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
!while/lstm_cell_2/strided_slice_3StridedSlice*while/lstm_cell_2/ReadVariableOp_3:value:00while/lstm_cell_2/strided_slice_3/stack:output:02while/lstm_cell_2/strided_slice_3/stack_1:output:02while/lstm_cell_2/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
while/lstm_cell_2/MatMul_7MatMulwhile_placeholder_2*while/lstm_cell_2/strided_slice_3:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_2/add_6AddV2$while/lstm_cell_2/BiasAdd_3:output:0$while/lstm_cell_2/MatMul_7:product:0*
T0*(
_output_shapes
:??????????^
while/lstm_cell_2/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>^
while/lstm_cell_2/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ??
while/lstm_cell_2/Mul_4Mulwhile/lstm_cell_2/add_6:z:0"while/lstm_cell_2/Const_4:output:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_2/Add_7AddV2while/lstm_cell_2/Mul_4:z:0"while/lstm_cell_2/Const_5:output:0*
T0*(
_output_shapes
:??????????p
+while/lstm_cell_2/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
)while/lstm_cell_2/clip_by_value_2/MinimumMinimumwhile/lstm_cell_2/Add_7:z:04while/lstm_cell_2/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:??????????h
#while/lstm_cell_2/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
!while/lstm_cell_2/clip_by_value_2Maximum-while/lstm_cell_2/clip_by_value_2/Minimum:z:0,while/lstm_cell_2/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:??????????p
while/lstm_cell_2/Tanh_1Tanhwhile/lstm_cell_2/add_5:z:0*
T0*(
_output_shapes
:???????????
while/lstm_cell_2/mul_5Mul%while/lstm_cell_2/clip_by_value_2:z:0while/lstm_cell_2/Tanh_1:y:0*
T0*(
_output_shapes
:???????????
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_2/mul_5:z:0*
_output_shapes
: *
element_dtype0:???M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: y
while/Identity_4Identitywhile/lstm_cell_2/mul_5:z:0^while/NoOp*
T0*(
_output_shapes
:??????????y
while/Identity_5Identitywhile/lstm_cell_2/add_5:z:0^while/NoOp*
T0*(
_output_shapes
:???????????

while/NoOpNoOp!^while/lstm_cell_2/ReadVariableOp#^while/lstm_cell_2/ReadVariableOp_1#^while/lstm_cell_2/ReadVariableOp_2#^while/lstm_cell_2/ReadVariableOp_3'^while/lstm_cell_2/split/ReadVariableOp)^while/lstm_cell_2/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"X
)while_lstm_cell_2_readvariableop_resource+while_lstm_cell_2_readvariableop_resource_0"h
1while_lstm_cell_2_split_1_readvariableop_resource3while_lstm_cell_2_split_1_readvariableop_resource_0"d
/while_lstm_cell_2_split_readvariableop_resource1while_lstm_cell_2_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2D
 while/lstm_cell_2/ReadVariableOp while/lstm_cell_2/ReadVariableOp2H
"while/lstm_cell_2/ReadVariableOp_1"while/lstm_cell_2/ReadVariableOp_12H
"while/lstm_cell_2/ReadVariableOp_2"while/lstm_cell_2/ReadVariableOp_22H
"while/lstm_cell_2/ReadVariableOp_3"while/lstm_cell_2/ReadVariableOp_32P
&while/lstm_cell_2/split/ReadVariableOp&while/lstm_cell_2/split/ReadVariableOp2T
(while/lstm_cell_2/split_1/ReadVariableOp(while/lstm_cell_2/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?
?
&__inference_lstm_2_layer_call_fn_58389

inputs
unknown:	?
	unknown_0:	?
	unknown_1:
??
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_lstm_2_layer_call_and_return_conditional_losses_56026t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?
A__inference_lstm_2_layer_call_and_return_conditional_losses_56026

inputs<
)lstm_cell_2_split_readvariableop_resource:	?:
+lstm_cell_2_split_1_readvariableop_resource:	?7
#lstm_cell_2_readvariableop_resource:
??
identity??lstm_cell_2/ReadVariableOp?lstm_cell_2/ReadVariableOp_1?lstm_cell_2/ReadVariableOp_2?lstm_cell_2/ReadVariableOp_3? lstm_cell_2/split/ReadVariableOp?"lstm_cell_2/split_1/ReadVariableOp?while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????S
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:?????????D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask]
lstm_cell_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
 lstm_cell_2/split/ReadVariableOpReadVariableOp)lstm_cell_2_split_readvariableop_resource*
_output_shapes
:	?*
dtype0?
lstm_cell_2/splitSplit$lstm_cell_2/split/split_dim:output:0(lstm_cell_2/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	?:	?:	?:	?*
	num_split?
lstm_cell_2/MatMulMatMulstrided_slice_2:output:0lstm_cell_2/split:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_2/MatMul_1MatMulstrided_slice_2:output:0lstm_cell_2/split:output:1*
T0*(
_output_shapes
:???????????
lstm_cell_2/MatMul_2MatMulstrided_slice_2:output:0lstm_cell_2/split:output:2*
T0*(
_output_shapes
:???????????
lstm_cell_2/MatMul_3MatMulstrided_slice_2:output:0lstm_cell_2/split:output:3*
T0*(
_output_shapes
:??????????_
lstm_cell_2/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
"lstm_cell_2/split_1/ReadVariableOpReadVariableOp+lstm_cell_2_split_1_readvariableop_resource*
_output_shapes	
:?*
dtype0?
lstm_cell_2/split_1Split&lstm_cell_2/split_1/split_dim:output:0*lstm_cell_2/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split?
lstm_cell_2/BiasAddBiasAddlstm_cell_2/MatMul:product:0lstm_cell_2/split_1:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_2/BiasAdd_1BiasAddlstm_cell_2/MatMul_1:product:0lstm_cell_2/split_1:output:1*
T0*(
_output_shapes
:???????????
lstm_cell_2/BiasAdd_2BiasAddlstm_cell_2/MatMul_2:product:0lstm_cell_2/split_1:output:2*
T0*(
_output_shapes
:???????????
lstm_cell_2/BiasAdd_3BiasAddlstm_cell_2/MatMul_3:product:0lstm_cell_2/split_1:output:3*
T0*(
_output_shapes
:???????????
lstm_cell_2/ReadVariableOpReadVariableOp#lstm_cell_2_readvariableop_resource* 
_output_shapes
:
??*
dtype0p
lstm_cell_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        r
!lstm_cell_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       r
!lstm_cell_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
lstm_cell_2/strided_sliceStridedSlice"lstm_cell_2/ReadVariableOp:value:0(lstm_cell_2/strided_slice/stack:output:0*lstm_cell_2/strided_slice/stack_1:output:0*lstm_cell_2/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
lstm_cell_2/MatMul_4MatMulzeros:output:0"lstm_cell_2/strided_slice:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_2/addAddV2lstm_cell_2/BiasAdd:output:0lstm_cell_2/MatMul_4:product:0*
T0*(
_output_shapes
:??????????V
lstm_cell_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>X
lstm_cell_2/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?z
lstm_cell_2/MulMullstm_cell_2/add:z:0lstm_cell_2/Const:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_2/Add_1AddV2lstm_cell_2/Mul:z:0lstm_cell_2/Const_1:output:0*
T0*(
_output_shapes
:??????????h
#lstm_cell_2/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
!lstm_cell_2/clip_by_value/MinimumMinimumlstm_cell_2/Add_1:z:0,lstm_cell_2/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:??????????`
lstm_cell_2/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
lstm_cell_2/clip_by_valueMaximum%lstm_cell_2/clip_by_value/Minimum:z:0$lstm_cell_2/clip_by_value/y:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_2/ReadVariableOp_1ReadVariableOp#lstm_cell_2_readvariableop_resource* 
_output_shapes
:
??*
dtype0r
!lstm_cell_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       t
#lstm_cell_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       t
#lstm_cell_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
lstm_cell_2/strided_slice_1StridedSlice$lstm_cell_2/ReadVariableOp_1:value:0*lstm_cell_2/strided_slice_1/stack:output:0,lstm_cell_2/strided_slice_1/stack_1:output:0,lstm_cell_2/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
lstm_cell_2/MatMul_5MatMulzeros:output:0$lstm_cell_2/strided_slice_1:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_2/add_2AddV2lstm_cell_2/BiasAdd_1:output:0lstm_cell_2/MatMul_5:product:0*
T0*(
_output_shapes
:??????????X
lstm_cell_2/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>X
lstm_cell_2/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ??
lstm_cell_2/Mul_1Mullstm_cell_2/add_2:z:0lstm_cell_2/Const_2:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_2/Add_3AddV2lstm_cell_2/Mul_1:z:0lstm_cell_2/Const_3:output:0*
T0*(
_output_shapes
:??????????j
%lstm_cell_2/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
#lstm_cell_2/clip_by_value_1/MinimumMinimumlstm_cell_2/Add_3:z:0.lstm_cell_2/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:??????????b
lstm_cell_2/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
lstm_cell_2/clip_by_value_1Maximum'lstm_cell_2/clip_by_value_1/Minimum:z:0&lstm_cell_2/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:??????????~
lstm_cell_2/mul_2Mullstm_cell_2/clip_by_value_1:z:0zeros_1:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_2/ReadVariableOp_2ReadVariableOp#lstm_cell_2_readvariableop_resource* 
_output_shapes
:
??*
dtype0r
!lstm_cell_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       t
#lstm_cell_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       t
#lstm_cell_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
lstm_cell_2/strided_slice_2StridedSlice$lstm_cell_2/ReadVariableOp_2:value:0*lstm_cell_2/strided_slice_2/stack:output:0,lstm_cell_2/strided_slice_2/stack_1:output:0,lstm_cell_2/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
lstm_cell_2/MatMul_6MatMulzeros:output:0$lstm_cell_2/strided_slice_2:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_2/add_4AddV2lstm_cell_2/BiasAdd_2:output:0lstm_cell_2/MatMul_6:product:0*
T0*(
_output_shapes
:??????????b
lstm_cell_2/TanhTanhlstm_cell_2/add_4:z:0*
T0*(
_output_shapes
:???????????
lstm_cell_2/mul_3Mullstm_cell_2/clip_by_value:z:0lstm_cell_2/Tanh:y:0*
T0*(
_output_shapes
:??????????{
lstm_cell_2/add_5AddV2lstm_cell_2/mul_2:z:0lstm_cell_2/mul_3:z:0*
T0*(
_output_shapes
:???????????
lstm_cell_2/ReadVariableOp_3ReadVariableOp#lstm_cell_2_readvariableop_resource* 
_output_shapes
:
??*
dtype0r
!lstm_cell_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       t
#lstm_cell_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        t
#lstm_cell_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
lstm_cell_2/strided_slice_3StridedSlice$lstm_cell_2/ReadVariableOp_3:value:0*lstm_cell_2/strided_slice_3/stack:output:0,lstm_cell_2/strided_slice_3/stack_1:output:0,lstm_cell_2/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask?
lstm_cell_2/MatMul_7MatMulzeros:output:0$lstm_cell_2/strided_slice_3:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_2/add_6AddV2lstm_cell_2/BiasAdd_3:output:0lstm_cell_2/MatMul_7:product:0*
T0*(
_output_shapes
:??????????X
lstm_cell_2/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>X
lstm_cell_2/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ??
lstm_cell_2/Mul_4Mullstm_cell_2/add_6:z:0lstm_cell_2/Const_4:output:0*
T0*(
_output_shapes
:???????????
lstm_cell_2/Add_7AddV2lstm_cell_2/Mul_4:z:0lstm_cell_2/Const_5:output:0*
T0*(
_output_shapes
:??????????j
%lstm_cell_2/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
#lstm_cell_2/clip_by_value_2/MinimumMinimumlstm_cell_2/Add_7:z:0.lstm_cell_2/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:??????????b
lstm_cell_2/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
lstm_cell_2/clip_by_value_2Maximum'lstm_cell_2/clip_by_value_2/Minimum:z:0&lstm_cell_2/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:??????????d
lstm_cell_2/Tanh_1Tanhlstm_cell_2/add_5:z:0*
T0*(
_output_shapes
:???????????
lstm_cell_2/mul_5Mullstm_cell_2/clip_by_value_2:z:0lstm_cell_2/Tanh_1:y:0*
T0*(
_output_shapes
:??????????n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)lstm_cell_2_split_readvariableop_resource+lstm_cell_2_split_1_readvariableop_resource#lstm_cell_2_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_55886*
condR
while_cond_55885*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????c
IdentityIdentitytranspose_1:y:0^NoOp*
T0*,
_output_shapes
:???????????
NoOpNoOp^lstm_cell_2/ReadVariableOp^lstm_cell_2/ReadVariableOp_1^lstm_cell_2/ReadVariableOp_2^lstm_cell_2/ReadVariableOp_3!^lstm_cell_2/split/ReadVariableOp#^lstm_cell_2/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????: : : 28
lstm_cell_2/ReadVariableOplstm_cell_2/ReadVariableOp2<
lstm_cell_2/ReadVariableOp_1lstm_cell_2/ReadVariableOp_12<
lstm_cell_2/ReadVariableOp_2lstm_cell_2/ReadVariableOp_22<
lstm_cell_2/ReadVariableOp_3lstm_cell_2/ReadVariableOp_32D
 lstm_cell_2/split/ReadVariableOp lstm_cell_2/split/ReadVariableOp2H
"lstm_cell_2/split_1/ReadVariableOp"lstm_cell_2/split_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
I
lstm_2_input9
serving_default_lstm_2_input:0?????????;
dense_30
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
	optimizer
regularization_losses
	variables
trainable_variables
		keras_api


signatures
*|&call_and_return_all_conditional_losses
}__call__
~_default_save_signature"
_tf_keras_sequential
?
cell

state_spec
regularization_losses
	variables
trainable_variables
	keras_api
*&call_and_return_all_conditional_losses
?__call__"
_tf_keras_rnn_layer
?
cell

state_spec
regularization_losses
	variables
trainable_variables
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_rnn_layer
?

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?

kernel
bias
regularization_losses
 	variables
!trainable_variables
"	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
#iter

$beta_1

%beta_2
	&decay
'learning_ratemhmimjmk(ml)mm*mn+mo,mp-mqvrvsvtvu(vv)vw*vx+vy,vz-v{"
tf_deprecated_optimizer
 "
trackable_list_wrapper
f
(0
)1
*2
+3
,4
-5
6
7
8
9"
trackable_list_wrapper
f
(0
)1
*2
+3
,4
-5
6
7
8
9"
trackable_list_wrapper
?
regularization_losses
.non_trainable_variables
	variables
/layer_regularization_losses
trainable_variables
0metrics
1layer_metrics

2layers
}__call__
~_default_save_signature
*|&call_and_return_all_conditional_losses
&|"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
?
3
state_size

(kernel
)recurrent_kernel
*bias
4regularization_losses
5	variables
6trainable_variables
7	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
(0
)1
*2"
trackable_list_wrapper
5
(0
)1
*2"
trackable_list_wrapper
?
regularization_losses
8non_trainable_variables
	variables
9layer_regularization_losses
trainable_variables
:metrics
;layer_metrics

<states

=layers
?__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
?
>
state_size

+kernel
,recurrent_kernel
-bias
?regularization_losses
@	variables
Atrainable_variables
B	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
+0
,1
-2"
trackable_list_wrapper
5
+0
,1
-2"
trackable_list_wrapper
?
regularization_losses
Cnon_trainable_variables
	variables
Dlayer_regularization_losses
trainable_variables
Emetrics
Flayer_metrics

Gstates

Hlayers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
": 
??2dense_2/kernel
:?2dense_2/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
regularization_losses
Inon_trainable_variables
	variables
Jlayer_regularization_losses
trainable_variables
Kmetrics
Llayer_metrics

Mlayers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
!:	?2dense_3/kernel
:2dense_3/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
regularization_losses
Nnon_trainable_variables
 	variables
Olayer_regularization_losses
!trainable_variables
Pmetrics
Qlayer_metrics

Rlayers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
,:*	?2lstm_2/lstm_cell_2/kernel
7:5
??2#lstm_2/lstm_cell_2/recurrent_kernel
&:$?2lstm_2/lstm_cell_2/bias
-:+
??2lstm_3/lstm_cell_3/kernel
7:5
??2#lstm_3/lstm_cell_3/recurrent_kernel
&:$?2lstm_3/lstm_cell_3/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
S0
T1"
trackable_list_wrapper
 "
trackable_dict_wrapper
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
(0
)1
*2"
trackable_list_wrapper
5
(0
)1
*2"
trackable_list_wrapper
?
4regularization_losses
Unon_trainable_variables
5	variables
Vlayer_regularization_losses
6trainable_variables
Wmetrics
Xlayer_metrics

Ylayers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
+0
,1
-2"
trackable_list_wrapper
5
+0
,1
-2"
trackable_list_wrapper
?
?regularization_losses
Znon_trainable_variables
@	variables
[layer_regularization_losses
Atrainable_variables
\metrics
]layer_metrics

^layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
N
	_total
	`count
a	variables
b	keras_api"
_tf_keras_metric
^
	ctotal
	dcount
e
_fn_kwargs
f	variables
g	keras_api"
_tf_keras_metric
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
:  (2total
:  (2count
.
_0
`1"
trackable_list_wrapper
-
a	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
c0
d1"
trackable_list_wrapper
-
f	variables"
_generic_user_object
':%
??2Adam/dense_2/kernel/m
 :?2Adam/dense_2/bias/m
&:$	?2Adam/dense_3/kernel/m
:2Adam/dense_3/bias/m
1:/	?2 Adam/lstm_2/lstm_cell_2/kernel/m
<::
??2*Adam/lstm_2/lstm_cell_2/recurrent_kernel/m
+:)?2Adam/lstm_2/lstm_cell_2/bias/m
2:0
??2 Adam/lstm_3/lstm_cell_3/kernel/m
<::
??2*Adam/lstm_3/lstm_cell_3/recurrent_kernel/m
+:)?2Adam/lstm_3/lstm_cell_3/bias/m
':%
??2Adam/dense_2/kernel/v
 :?2Adam/dense_2/bias/v
&:$	?2Adam/dense_3/kernel/v
:2Adam/dense_3/bias/v
1:/	?2 Adam/lstm_2/lstm_cell_2/kernel/v
<::
??2*Adam/lstm_2/lstm_cell_2/recurrent_kernel/v
+:)?2Adam/lstm_2/lstm_cell_2/bias/v
2:0
??2 Adam/lstm_3/lstm_cell_3/kernel/v
<::
??2*Adam/lstm_3/lstm_cell_3/recurrent_kernel/v
+:)?2Adam/lstm_3/lstm_cell_3/bias/v
?2?
G__inference_sequential_1_layer_call_and_return_conditional_losses_56749
G__inference_sequential_1_layer_call_and_return_conditional_losses_57271
G__inference_sequential_1_layer_call_and_return_conditional_losses_56166
G__inference_sequential_1_layer_call_and_return_conditional_losses_56194?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
,__inference_sequential_1_layer_call_fn_55459
,__inference_sequential_1_layer_call_fn_57296
,__inference_sequential_1_layer_call_fn_57321
,__inference_sequential_1_layer_call_fn_56138?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
 __inference__wrapped_model_53944?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? */?,
*?'
lstm_2_input?????????
?2?
A__inference_lstm_2_layer_call_and_return_conditional_losses_57577
A__inference_lstm_2_layer_call_and_return_conditional_losses_57833
A__inference_lstm_2_layer_call_and_return_conditional_losses_58089
A__inference_lstm_2_layer_call_and_return_conditional_losses_58345?
???
FullArgSpecB
args:?7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults?

 
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
&__inference_lstm_2_layer_call_fn_58356
&__inference_lstm_2_layer_call_fn_58367
&__inference_lstm_2_layer_call_fn_58378
&__inference_lstm_2_layer_call_fn_58389?
???
FullArgSpecB
args:?7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults?

 
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
A__inference_lstm_3_layer_call_and_return_conditional_losses_58645
A__inference_lstm_3_layer_call_and_return_conditional_losses_58901
A__inference_lstm_3_layer_call_and_return_conditional_losses_59157
A__inference_lstm_3_layer_call_and_return_conditional_losses_59413?
???
FullArgSpecB
args:?7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults?

 
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
&__inference_lstm_3_layer_call_fn_59424
&__inference_lstm_3_layer_call_fn_59435
&__inference_lstm_3_layer_call_fn_59446
&__inference_lstm_3_layer_call_fn_59457?
???
FullArgSpecB
args:?7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults?

 
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
B__inference_dense_2_layer_call_and_return_conditional_losses_59468?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_dense_2_layer_call_fn_59477?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_dense_3_layer_call_and_return_conditional_losses_59488?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_dense_3_layer_call_fn_59497?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
#__inference_signature_wrapper_56227lstm_2_input"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_lstm_cell_2_layer_call_and_return_conditional_losses_59586
F__inference_lstm_cell_2_layer_call_and_return_conditional_losses_59675?
???
FullArgSpec3
args+?(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
+__inference_lstm_cell_2_layer_call_fn_59692
+__inference_lstm_cell_2_layer_call_fn_59709?
???
FullArgSpec3
args+?(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
F__inference_lstm_cell_3_layer_call_and_return_conditional_losses_59798
F__inference_lstm_cell_3_layer_call_and_return_conditional_losses_59887?
???
FullArgSpec3
args+?(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
+__inference_lstm_cell_3_layer_call_fn_59904
+__inference_lstm_cell_3_layer_call_fn_59921?
???
FullArgSpec3
args+?(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 ?
 __inference__wrapped_model_53944z
(*)+-,9?6
/?,
*?'
lstm_2_input?????????
? "1?.
,
dense_3!?
dense_3??????????
B__inference_dense_2_layer_call_and_return_conditional_losses_59468^0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? |
'__inference_dense_2_layer_call_fn_59477Q0?-
&?#
!?
inputs??????????
? "????????????
B__inference_dense_3_layer_call_and_return_conditional_losses_59488]0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????
? {
'__inference_dense_3_layer_call_fn_59497P0?-
&?#
!?
inputs??????????
? "???????????
A__inference_lstm_2_layer_call_and_return_conditional_losses_57577?(*)O?L
E?B
4?1
/?,
inputs/0??????????????????

 
p 

 
? "3?0
)?&
0???????????????????
? ?
A__inference_lstm_2_layer_call_and_return_conditional_losses_57833?(*)O?L
E?B
4?1
/?,
inputs/0??????????????????

 
p

 
? "3?0
)?&
0???????????????????
? ?
A__inference_lstm_2_layer_call_and_return_conditional_losses_58089r(*)??<
5?2
$?!
inputs?????????

 
p 

 
? "*?'
 ?
0??????????
? ?
A__inference_lstm_2_layer_call_and_return_conditional_losses_58345r(*)??<
5?2
$?!
inputs?????????

 
p

 
? "*?'
 ?
0??????????
? ?
&__inference_lstm_2_layer_call_fn_58356~(*)O?L
E?B
4?1
/?,
inputs/0??????????????????

 
p 

 
? "&?#????????????????????
&__inference_lstm_2_layer_call_fn_58367~(*)O?L
E?B
4?1
/?,
inputs/0??????????????????

 
p

 
? "&?#????????????????????
&__inference_lstm_2_layer_call_fn_58378e(*)??<
5?2
$?!
inputs?????????

 
p 

 
? "????????????
&__inference_lstm_2_layer_call_fn_58389e(*)??<
5?2
$?!
inputs?????????

 
p

 
? "????????????
A__inference_lstm_3_layer_call_and_return_conditional_losses_58645+-,P?M
F?C
5?2
0?-
inputs/0???????????????????

 
p 

 
? "&?#
?
0??????????
? ?
A__inference_lstm_3_layer_call_and_return_conditional_losses_58901+-,P?M
F?C
5?2
0?-
inputs/0???????????????????

 
p

 
? "&?#
?
0??????????
? ?
A__inference_lstm_3_layer_call_and_return_conditional_losses_59157o+-,@?=
6?3
%?"
inputs??????????

 
p 

 
? "&?#
?
0??????????
? ?
A__inference_lstm_3_layer_call_and_return_conditional_losses_59413o+-,@?=
6?3
%?"
inputs??????????

 
p

 
? "&?#
?
0??????????
? ?
&__inference_lstm_3_layer_call_fn_59424r+-,P?M
F?C
5?2
0?-
inputs/0???????????????????

 
p 

 
? "????????????
&__inference_lstm_3_layer_call_fn_59435r+-,P?M
F?C
5?2
0?-
inputs/0???????????????????

 
p

 
? "????????????
&__inference_lstm_3_layer_call_fn_59446b+-,@?=
6?3
%?"
inputs??????????

 
p 

 
? "????????????
&__inference_lstm_3_layer_call_fn_59457b+-,@?=
6?3
%?"
inputs??????????

 
p

 
? "????????????
F__inference_lstm_cell_2_layer_call_and_return_conditional_losses_59586?(*)??
x?u
 ?
inputs?????????
M?J
#? 
states/0??????????
#? 
states/1??????????
p 
? "v?s
l?i
?
0/0??????????
G?D
 ?
0/1/0??????????
 ?
0/1/1??????????
? ?
F__inference_lstm_cell_2_layer_call_and_return_conditional_losses_59675?(*)??
x?u
 ?
inputs?????????
M?J
#? 
states/0??????????
#? 
states/1??????????
p
? "v?s
l?i
?
0/0??????????
G?D
 ?
0/1/0??????????
 ?
0/1/1??????????
? ?
+__inference_lstm_cell_2_layer_call_fn_59692?(*)??
x?u
 ?
inputs?????????
M?J
#? 
states/0??????????
#? 
states/1??????????
p 
? "f?c
?
0??????????
C?@
?
1/0??????????
?
1/1???????????
+__inference_lstm_cell_2_layer_call_fn_59709?(*)??
x?u
 ?
inputs?????????
M?J
#? 
states/0??????????
#? 
states/1??????????
p
? "f?c
?
0??????????
C?@
?
1/0??????????
?
1/1???????????
F__inference_lstm_cell_3_layer_call_and_return_conditional_losses_59798?+-,???
y?v
!?
inputs??????????
M?J
#? 
states/0??????????
#? 
states/1??????????
p 
? "v?s
l?i
?
0/0??????????
G?D
 ?
0/1/0??????????
 ?
0/1/1??????????
? ?
F__inference_lstm_cell_3_layer_call_and_return_conditional_losses_59887?+-,???
y?v
!?
inputs??????????
M?J
#? 
states/0??????????
#? 
states/1??????????
p
? "v?s
l?i
?
0/0??????????
G?D
 ?
0/1/0??????????
 ?
0/1/1??????????
? ?
+__inference_lstm_cell_3_layer_call_fn_59904?+-,???
y?v
!?
inputs??????????
M?J
#? 
states/0??????????
#? 
states/1??????????
p 
? "f?c
?
0??????????
C?@
?
1/0??????????
?
1/1???????????
+__inference_lstm_cell_3_layer_call_fn_59921?+-,???
y?v
!?
inputs??????????
M?J
#? 
states/0??????????
#? 
states/1??????????
p
? "f?c
?
0??????????
C?@
?
1/0??????????
?
1/1???????????
G__inference_sequential_1_layer_call_and_return_conditional_losses_56166v
(*)+-,A?>
7?4
*?'
lstm_2_input?????????
p 

 
? "%?"
?
0?????????
? ?
G__inference_sequential_1_layer_call_and_return_conditional_losses_56194v
(*)+-,A?>
7?4
*?'
lstm_2_input?????????
p

 
? "%?"
?
0?????????
? ?
G__inference_sequential_1_layer_call_and_return_conditional_losses_56749p
(*)+-,;?8
1?.
$?!
inputs?????????
p 

 
? "%?"
?
0?????????
? ?
G__inference_sequential_1_layer_call_and_return_conditional_losses_57271p
(*)+-,;?8
1?.
$?!
inputs?????????
p

 
? "%?"
?
0?????????
? ?
,__inference_sequential_1_layer_call_fn_55459i
(*)+-,A?>
7?4
*?'
lstm_2_input?????????
p 

 
? "???????????
,__inference_sequential_1_layer_call_fn_56138i
(*)+-,A?>
7?4
*?'
lstm_2_input?????????
p

 
? "???????????
,__inference_sequential_1_layer_call_fn_57296c
(*)+-,;?8
1?.
$?!
inputs?????????
p 

 
? "???????????
,__inference_sequential_1_layer_call_fn_57321c
(*)+-,;?8
1?.
$?!
inputs?????????
p

 
? "???????????
#__inference_signature_wrapper_56227?
(*)+-,I?F
? 
??<
:
lstm_2_input*?'
lstm_2_input?????????"1?.
,
dense_3!?
dense_3?????????