��
��
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
�
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
executor_typestring �
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape�"serve*2.2.02v2.2.0-rc4-8-g2b96f3662b8��
t
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:!!*
shared_namedense/kernel
m
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes

:!!*
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:!*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:!*
dtype0
n
	pi/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:!d*
shared_name	pi/kernel
g
pi/kernel/Read/ReadVariableOpReadVariableOp	pi/kernel*
_output_shapes

:!d*
dtype0
f
pi/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*
shared_name	pi/bias
_
pi/bias/Read/ReadVariableOpReadVariableOppi/bias*
_output_shapes
:d*
dtype0
l
v/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:!*
shared_name
v/kernel
e
v/kernel/Read/ReadVariableOpReadVariableOpv/kernel*
_output_shapes

:!*
dtype0
d
v/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namev/bias
]
v/bias/Read/ReadVariableOpReadVariableOpv/bias*
_output_shapes
:*
dtype0
^
decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedecay
W
decay/Read/ReadVariableOpReadVariableOpdecay*
_output_shapes
: *
dtype0
n
learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
d
momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
momentum
]
momentum/Read/ReadVariableOpReadVariableOpmomentum*
_output_shapes
: *
dtype0

NoOpNoOp
�
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�
value�B� B�
�
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
	optimizer
loss
regularization_losses
	trainable_variables

	variables
	keras_api

signatures
 
h

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
R
regularization_losses
trainable_variables
	variables
	keras_api
h

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
h

kernel
bias
regularization_losses
 trainable_variables
!	variables
"	keras_api
,
	#decay
$learning_rate
%momentum
 
 
*
0
1
2
3
4
5
*
0
1
2
3
4
5
�
&metrics
'non_trainable_variables

(layers
regularization_losses
)layer_regularization_losses
	trainable_variables

	variables
*layer_metrics
 
XV
VARIABLE_VALUEdense/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
dense/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
�
+metrics
,non_trainable_variables

-layers
regularization_losses
.layer_regularization_losses
trainable_variables
	variables
/layer_metrics
 
 
 
�
0metrics
1non_trainable_variables

2layers
regularization_losses
3layer_regularization_losses
trainable_variables
	variables
4layer_metrics
US
VARIABLE_VALUE	pi/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEpi/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
�
5metrics
6non_trainable_variables

7layers
regularization_losses
8layer_regularization_losses
trainable_variables
	variables
9layer_metrics
TR
VARIABLE_VALUEv/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
PN
VARIABLE_VALUEv/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
�
:metrics
;non_trainable_variables

<layers
regularization_losses
=layer_regularization_losses
 trainable_variables
!	variables
>layer_metrics
EC
VARIABLE_VALUEdecay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUElearning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEmomentum-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
 
 
#
0
1
2
3
4
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
 
 
z
serving_default_input_1Placeholder*'
_output_shapes
:���������!*
dtype0*
shape:���������!
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1dense/kernel
dense/biasv/kernelv/bias	pi/kernelpi/bias*
Tin
	2*
Tout
2*:
_output_shapes(
&:���������d:���������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU2*0J 8**
f%R#
!__inference_signature_wrapper_427
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOppi/kernel/Read/ReadVariableOppi/bias/Read/ReadVariableOpv/kernel/Read/ReadVariableOpv/bias/Read/ReadVariableOpdecay/Read/ReadVariableOp!learning_rate/Read/ReadVariableOpmomentum/Read/ReadVariableOpConst*
Tin
2*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*%
f R
__inference__traced_save_692
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense/kernel
dense/bias	pi/kernelpi/biasv/kernelv/biasdecaylearning_ratemomentum*
Tin
2
*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*(
f#R!
__inference__traced_restore_731��
�

_
@__inference_dropout_layer_call_and_return_conditional_losses_219

inputs
identity�c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:���������!2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:���������!*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dropout/GreaterEqual/y�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������!2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:���������!2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:���������!2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:���������!2

Identity"
identityIdentity:output:0*&
_input_shapes
:���������!:O K
'
_output_shapes
:���������!
 
_user_specified_nameinputs
�
�
>__inference_model_layer_call_and_return_conditional_losses_500

inputs(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource$
 v_matmul_readvariableop_resource%
!v_biasadd_readvariableop_resource%
!pi_matmul_readvariableop_resource&
"pi_biasadd_readvariableop_resource
identity

identity_1��
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:!!*
dtype02
dense/MatMul/ReadVariableOp�
dense/MatMulMatMulinputs#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������!2
dense/MatMul�
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:!*
dtype02
dense/BiasAdd/ReadVariableOp�
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������!2
dense/BiasAdds
dense/SigmoidSigmoiddense/BiasAdd:output:0*
T0*'
_output_shapes
:���������!2
dense/Sigmoidz
	dense/mulMuldense/BiasAdd:output:0dense/Sigmoid:y:0*
T0*'
_output_shapes
:���������!2
	dense/mulm
dense/IdentityIdentitydense/mul:z:0*
T0*'
_output_shapes
:���������!2
dense/Identity�
dense/IdentityN	IdentityNdense/mul:z:0dense/BiasAdd:output:0*
T
2*)
_gradient_op_typeCustomGradient-476*:
_output_shapes(
&:���������!:���������!2
dense/IdentityN|
dropout/IdentityIdentitydense/IdentityN:output:0*
T0*'
_output_shapes
:���������!2
dropout/Identity�
v/MatMul/ReadVariableOpReadVariableOp v_matmul_readvariableop_resource*
_output_shapes

:!*
dtype02
v/MatMul/ReadVariableOp�
v/MatMulMatMuldropout/Identity:output:0v/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2

v/MatMul�
v/BiasAdd/ReadVariableOpReadVariableOp!v_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
v/BiasAdd/ReadVariableOp�
	v/BiasAddBiasAddv/MatMul:product:0 v/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
	v/BiasAdd^
v/TanhTanhv/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
v/Tanh�
pi/MatMul/ReadVariableOpReadVariableOp!pi_matmul_readvariableop_resource*
_output_shapes

:!d*
dtype02
pi/MatMul/ReadVariableOp�
	pi/MatMulMatMuldropout/Identity:output:0 pi/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d2
	pi/MatMul�
pi/BiasAdd/ReadVariableOpReadVariableOp"pi_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02
pi/BiasAdd/ReadVariableOp�

pi/BiasAddBiasAddpi/MatMul:product:0!pi/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d2

pi/BiasAddj

pi/SoftmaxSoftmaxpi/BiasAdd:output:0*
T0*'
_output_shapes
:���������d2

pi/Softmax�
dense/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense/kernel/Regularizer/Consth
IdentityIdentitypi/Softmax:softmax:0*
T0*'
_output_shapes
:���������d2

Identityb

Identity_1Identity
v/Tanh:y:0*
T0*'
_output_shapes
:���������2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*>
_input_shapes-
+:���������!:::::::O K
'
_output_shapes
:���������!
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�

�
#__inference_model_layer_call_fn_399
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*:
_output_shapes(
&:���������d:���������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU2*0J 8*G
fBR@
>__inference_model_layer_call_and_return_conditional_losses_3822
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������d2

Identity�

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*>
_input_shapes-
+:���������!::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:���������!
!
_user_specified_name	input_1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
A
%__inference_dropout_layer_call_fn_592

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*'
_output_shapes
:���������!* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*I
fDRB
@__inference_dropout_layer_call_and_return_conditional_losses_2242
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������!2

Identity"
identityIdentity:output:0*&
_input_shapes
:���������!:O K
'
_output_shapes
:���������!
 
_user_specified_nameinputs
�
�
;__inference_pi_layer_call_and_return_conditional_losses_275

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:!d*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:���������d2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:���������d2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������!:::O K
'
_output_shapes
:���������!
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�
x
#__inference_dense_layer_call_fn_565

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*'
_output_shapes
:���������!*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*G
fBR@
>__inference_dense_layer_call_and_return_conditional_losses_1912
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������!2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������!::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������!
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�
�
:__inference_v_layer_call_and_return_conditional_losses_623

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:!*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:���������2
Tanh\
IdentityIdentityTanh:y:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������!:::O K
'
_output_shapes
:���������!
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�

�
#__inference_model_layer_call_fn_358
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*:
_output_shapes(
&:���������d:���������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU2*0J 8*G
fBR@
>__inference_model_layer_call_and_return_conditional_losses_3412
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������d2

Identity�

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*>
_input_shapes-
+:���������!::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:���������!
!
_user_specified_name	input_1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
�
>__inference_dense_layer_call_and_return_conditional_losses_556

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource

identity_1��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:!!*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������!2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:!*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������!2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:���������!2	
Sigmoidb
mulMulBiasAdd:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������!2
mul[
IdentityIdentitymul:z:0*
T0*'
_output_shapes
:���������!2

Identity�
	IdentityN	IdentityNmul:z:0BiasAdd:output:0*
T
2*)
_gradient_op_typeCustomGradient-548*:
_output_shapes(
&:���������!:���������!2
	IdentityN�
dense/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense/kernel/Regularizer/Constj

Identity_1IdentityIdentityN:output:0*
T0*'
_output_shapes
:���������!2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:���������!:::O K
'
_output_shapes
:���������!
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�
u
 __inference_pi_layer_call_fn_612

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*D
f?R=
;__inference_pi_layer_call_and_return_conditional_losses_2752
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������d2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������!::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������!
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�
�
__inference__wrapped_model_170
input_1.
*model_dense_matmul_readvariableop_resource/
+model_dense_biasadd_readvariableop_resource*
&model_v_matmul_readvariableop_resource+
'model_v_biasadd_readvariableop_resource+
'model_pi_matmul_readvariableop_resource,
(model_pi_biasadd_readvariableop_resource
identity

identity_1��
!model/dense/MatMul/ReadVariableOpReadVariableOp*model_dense_matmul_readvariableop_resource*
_output_shapes

:!!*
dtype02#
!model/dense/MatMul/ReadVariableOp�
model/dense/MatMulMatMulinput_1)model/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������!2
model/dense/MatMul�
"model/dense/BiasAdd/ReadVariableOpReadVariableOp+model_dense_biasadd_readvariableop_resource*
_output_shapes
:!*
dtype02$
"model/dense/BiasAdd/ReadVariableOp�
model/dense/BiasAddBiasAddmodel/dense/MatMul:product:0*model/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������!2
model/dense/BiasAdd�
model/dense/SigmoidSigmoidmodel/dense/BiasAdd:output:0*
T0*'
_output_shapes
:���������!2
model/dense/Sigmoid�
model/dense/mulMulmodel/dense/BiasAdd:output:0model/dense/Sigmoid:y:0*
T0*'
_output_shapes
:���������!2
model/dense/mul
model/dense/IdentityIdentitymodel/dense/mul:z:0*
T0*'
_output_shapes
:���������!2
model/dense/Identity�
model/dense/IdentityN	IdentityNmodel/dense/mul:z:0model/dense/BiasAdd:output:0*
T
2*)
_gradient_op_typeCustomGradient-147*:
_output_shapes(
&:���������!:���������!2
model/dense/IdentityN�
model/dropout/IdentityIdentitymodel/dense/IdentityN:output:0*
T0*'
_output_shapes
:���������!2
model/dropout/Identity�
model/v/MatMul/ReadVariableOpReadVariableOp&model_v_matmul_readvariableop_resource*
_output_shapes

:!*
dtype02
model/v/MatMul/ReadVariableOp�
model/v/MatMulMatMulmodel/dropout/Identity:output:0%model/v/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
model/v/MatMul�
model/v/BiasAdd/ReadVariableOpReadVariableOp'model_v_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
model/v/BiasAdd/ReadVariableOp�
model/v/BiasAddBiasAddmodel/v/MatMul:product:0&model/v/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
model/v/BiasAddp
model/v/TanhTanhmodel/v/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
model/v/Tanh�
model/pi/MatMul/ReadVariableOpReadVariableOp'model_pi_matmul_readvariableop_resource*
_output_shapes

:!d*
dtype02 
model/pi/MatMul/ReadVariableOp�
model/pi/MatMulMatMulmodel/dropout/Identity:output:0&model/pi/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d2
model/pi/MatMul�
model/pi/BiasAdd/ReadVariableOpReadVariableOp(model_pi_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02!
model/pi/BiasAdd/ReadVariableOp�
model/pi/BiasAddBiasAddmodel/pi/MatMul:product:0'model/pi/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d2
model/pi/BiasAdd|
model/pi/SoftmaxSoftmaxmodel/pi/BiasAdd:output:0*
T0*'
_output_shapes
:���������d2
model/pi/Softmaxn
IdentityIdentitymodel/pi/Softmax:softmax:0*
T0*'
_output_shapes
:���������d2

Identityh

Identity_1Identitymodel/v/Tanh:y:0*
T0*'
_output_shapes
:���������2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*>
_input_shapes-
+:���������!:::::::P L
'
_output_shapes
:���������!
!
_user_specified_name	input_1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�&
�
>__inference_model_layer_call_and_return_conditional_losses_467

inputs(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource$
 v_matmul_readvariableop_resource%
!v_biasadd_readvariableop_resource%
!pi_matmul_readvariableop_resource&
"pi_biasadd_readvariableop_resource
identity

identity_1��
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:!!*
dtype02
dense/MatMul/ReadVariableOp�
dense/MatMulMatMulinputs#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������!2
dense/MatMul�
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:!*
dtype02
dense/BiasAdd/ReadVariableOp�
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������!2
dense/BiasAdds
dense/SigmoidSigmoiddense/BiasAdd:output:0*
T0*'
_output_shapes
:���������!2
dense/Sigmoidz
	dense/mulMuldense/BiasAdd:output:0dense/Sigmoid:y:0*
T0*'
_output_shapes
:���������!2
	dense/mulm
dense/IdentityIdentitydense/mul:z:0*
T0*'
_output_shapes
:���������!2
dense/Identity�
dense/IdentityN	IdentityNdense/mul:z:0dense/BiasAdd:output:0*
T
2*)
_gradient_op_typeCustomGradient-436*:
_output_shapes(
&:���������!:���������!2
dense/IdentityNs
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
dropout/dropout/Const�
dropout/dropout/MulMuldense/IdentityN:output:0dropout/dropout/Const:output:0*
T0*'
_output_shapes
:���������!2
dropout/dropout/Mulv
dropout/dropout/ShapeShapedense/IdentityN:output:0*
T0*
_output_shapes
:2
dropout/dropout/Shape�
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*'
_output_shapes
:���������!*
dtype02.
,dropout/dropout/random_uniform/RandomUniform�
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dropout/dropout/GreaterEqual/y�
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������!2
dropout/dropout/GreaterEqual�
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:���������!2
dropout/dropout/Cast�
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*'
_output_shapes
:���������!2
dropout/dropout/Mul_1�
v/MatMul/ReadVariableOpReadVariableOp v_matmul_readvariableop_resource*
_output_shapes

:!*
dtype02
v/MatMul/ReadVariableOp�
v/MatMulMatMuldropout/dropout/Mul_1:z:0v/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2

v/MatMul�
v/BiasAdd/ReadVariableOpReadVariableOp!v_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
v/BiasAdd/ReadVariableOp�
	v/BiasAddBiasAddv/MatMul:product:0 v/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
	v/BiasAdd^
v/TanhTanhv/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
v/Tanh�
pi/MatMul/ReadVariableOpReadVariableOp!pi_matmul_readvariableop_resource*
_output_shapes

:!d*
dtype02
pi/MatMul/ReadVariableOp�
	pi/MatMulMatMuldropout/dropout/Mul_1:z:0 pi/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d2
	pi/MatMul�
pi/BiasAdd/ReadVariableOpReadVariableOp"pi_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02
pi/BiasAdd/ReadVariableOp�

pi/BiasAddBiasAddpi/MatMul:product:0!pi/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d2

pi/BiasAddj

pi/SoftmaxSoftmaxpi/BiasAdd:output:0*
T0*'
_output_shapes
:���������d2

pi/Softmax�
dense/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense/kernel/Regularizer/Consth
IdentityIdentitypi/Softmax:softmax:0*
T0*'
_output_shapes
:���������d2

Identityb

Identity_1Identity
v/Tanh:y:0*
T0*'
_output_shapes
:���������2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*>
_input_shapes-
+:���������!:::::::O K
'
_output_shapes
:���������!
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
^
@__inference_dropout_layer_call_and_return_conditional_losses_582

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:���������!2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:���������!2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:���������!:O K
'
_output_shapes
:���������!
 
_user_specified_nameinputs
�
�
>__inference_model_layer_call_and_return_conditional_losses_294
input_1
	dense_202
	dense_204	
v_259	
v_261

pi_286

pi_288
identity

identity_1��dense/StatefulPartitionedCall�dropout/StatefulPartitionedCall�pi/StatefulPartitionedCall�v/StatefulPartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCallinput_1	dense_202	dense_204*
Tin
2*
Tout
2*'
_output_shapes
:���������!*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*G
fBR@
>__inference_dense_layer_call_and_return_conditional_losses_1912
dense/StatefulPartitionedCall�
dropout/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:���������!* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*I
fDRB
@__inference_dropout_layer_call_and_return_conditional_losses_2192!
dropout/StatefulPartitionedCall�
v/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0v_259v_261*
Tin
2*
Tout
2*'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*C
f>R<
:__inference_v_layer_call_and_return_conditional_losses_2482
v/StatefulPartitionedCall�
pi/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0pi_286pi_288*
Tin
2*
Tout
2*'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*D
f?R=
;__inference_pi_layer_call_and_return_conditional_losses_2752
pi/StatefulPartitionedCall�
dense/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense/kernel/Regularizer/Const�
IdentityIdentity#pi/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dropout/StatefulPartitionedCall^pi/StatefulPartitionedCall^v/StatefulPartitionedCall*
T0*'
_output_shapes
:���������d2

Identity�

Identity_1Identity"v/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dropout/StatefulPartitionedCall^pi/StatefulPartitionedCall^v/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*>
_input_shapes-
+:���������!::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall28
pi/StatefulPartitionedCallpi/StatefulPartitionedCall26
v/StatefulPartitionedCallv/StatefulPartitionedCall:P L
'
_output_shapes
:���������!
!
_user_specified_name	input_1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�&
�
__inference__traced_save_692
file_prefix+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop(
$savev2_pi_kernel_read_readvariableop&
"savev2_pi_bias_read_readvariableop'
#savev2_v_kernel_read_readvariableop%
!savev2_v_bias_read_readvariableop$
 savev2_decay_read_readvariableop,
(savev2_learning_rate_read_readvariableop'
#savev2_momentum_read_readvariableop
savev2_1_const

identity_1��MergeV2Checkpoints�SaveV2�SaveV2_1�
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Const�
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_72bd39540e064e9c871d2d0adb2f0683/part2	
Const_1�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard�
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename�
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*�
value�B�	B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*%
valueB	B B B B B B B B B 2
SaveV2/shape_and_slices�
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop$savev2_pi_kernel_read_readvariableop"savev2_pi_bias_read_readvariableop#savev2_v_kernel_read_readvariableop!savev2_v_bias_read_readvariableop savev2_decay_read_readvariableop(savev2_learning_rate_read_readvariableop#savev2_momentum_read_readvariableop"/device:CPU:0*
_output_shapes
 *
dtypes
2	2
SaveV2�
ShardedFilename_1/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B :2
ShardedFilename_1/shard�
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename_1�
SaveV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2_1/tensor_names�
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
SaveV2_1/shape_and_slices�
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
22

SaveV2_1�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity�

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*M
_input_shapes<
:: :!!:!:!d:d:!:: : : : 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:!!: 

_output_shapes
:!:$ 

_output_shapes

:!d: 

_output_shapes
:d:$ 

_output_shapes

:!: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: 
�

_
@__inference_dropout_layer_call_and_return_conditional_losses_577

inputs
identity�c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:���������!2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:���������!*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dropout/GreaterEqual/y�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������!2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:���������!2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:���������!2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:���������!2

Identity"
identityIdentity:output:0*&
_input_shapes
:���������!:O K
'
_output_shapes
:���������!
 
_user_specified_nameinputs
�
�
;__inference_pi_layer_call_and_return_conditional_losses_603

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:!d*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:���������d2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:���������d2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������!:::O K
'
_output_shapes
:���������!
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�
�
:__inference_v_layer_call_and_return_conditional_losses_248

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:!*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:���������2
Tanh\
IdentityIdentityTanh:y:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������!:::O K
'
_output_shapes
:���������!
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�
�
>__inference_dense_layer_call_and_return_conditional_losses_191

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource

identity_1��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:!!*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������!2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:!*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������!2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:���������!2	
Sigmoidb
mulMulBiasAdd:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������!2
mul[
IdentityIdentitymul:z:0*
T0*'
_output_shapes
:���������!2

Identity�
	IdentityN	IdentityNmul:z:0BiasAdd:output:0*
T
2*)
_gradient_op_typeCustomGradient-183*:
_output_shapes(
&:���������!:���������!2
	IdentityN�
dense/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense/kernel/Regularizer/Constj

Identity_1IdentityIdentityN:output:0*
T0*'
_output_shapes
:���������!2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:���������!:::O K
'
_output_shapes
:���������!
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�
)
__inference_loss_fn_0_637
identity�
dense/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense/kernel/Regularizer/Constj
IdentityIdentity'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes 
�
t
__inference_v_layer_call_fn_632

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*C
f>R<
:__inference_v_layer_call_and_return_conditional_losses_2482
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������!::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������!
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�
�
>__inference_model_layer_call_and_return_conditional_losses_382

inputs
	dense_363
	dense_365	
v_369	
v_371

pi_374

pi_376
identity

identity_1��dense/StatefulPartitionedCall�pi/StatefulPartitionedCall�v/StatefulPartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCallinputs	dense_363	dense_365*
Tin
2*
Tout
2*'
_output_shapes
:���������!*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*G
fBR@
>__inference_dense_layer_call_and_return_conditional_losses_1912
dense/StatefulPartitionedCall�
dropout/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:���������!* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*I
fDRB
@__inference_dropout_layer_call_and_return_conditional_losses_2242
dropout/PartitionedCall�
v/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0v_369v_371*
Tin
2*
Tout
2*'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*C
f>R<
:__inference_v_layer_call_and_return_conditional_losses_2482
v/StatefulPartitionedCall�
pi/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0pi_374pi_376*
Tin
2*
Tout
2*'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*D
f?R=
;__inference_pi_layer_call_and_return_conditional_losses_2752
pi/StatefulPartitionedCall�
dense/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense/kernel/Regularizer/Const�
IdentityIdentity#pi/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall^pi/StatefulPartitionedCall^v/StatefulPartitionedCall*
T0*'
_output_shapes
:���������d2

Identity�

Identity_1Identity"v/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall^pi/StatefulPartitionedCall^v/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*>
_input_shapes-
+:���������!::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall28
pi/StatefulPartitionedCallpi/StatefulPartitionedCall26
v/StatefulPartitionedCallv/StatefulPartitionedCall:O K
'
_output_shapes
:���������!
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
�
>__inference_model_layer_call_and_return_conditional_losses_341

inputs
	dense_322
	dense_324	
v_328	
v_330

pi_333

pi_335
identity

identity_1��dense/StatefulPartitionedCall�dropout/StatefulPartitionedCall�pi/StatefulPartitionedCall�v/StatefulPartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCallinputs	dense_322	dense_324*
Tin
2*
Tout
2*'
_output_shapes
:���������!*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*G
fBR@
>__inference_dense_layer_call_and_return_conditional_losses_1912
dense/StatefulPartitionedCall�
dropout/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:���������!* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*I
fDRB
@__inference_dropout_layer_call_and_return_conditional_losses_2192!
dropout/StatefulPartitionedCall�
v/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0v_328v_330*
Tin
2*
Tout
2*'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*C
f>R<
:__inference_v_layer_call_and_return_conditional_losses_2482
v/StatefulPartitionedCall�
pi/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0pi_333pi_335*
Tin
2*
Tout
2*'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*D
f?R=
;__inference_pi_layer_call_and_return_conditional_losses_2752
pi/StatefulPartitionedCall�
dense/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense/kernel/Regularizer/Const�
IdentityIdentity#pi/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dropout/StatefulPartitionedCall^pi/StatefulPartitionedCall^v/StatefulPartitionedCall*
T0*'
_output_shapes
:���������d2

Identity�

Identity_1Identity"v/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dropout/StatefulPartitionedCall^pi/StatefulPartitionedCall^v/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*>
_input_shapes-
+:���������!::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall28
pi/StatefulPartitionedCallpi/StatefulPartitionedCall26
v/StatefulPartitionedCallv/StatefulPartitionedCall:O K
'
_output_shapes
:���������!
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
^
@__inference_dropout_layer_call_and_return_conditional_losses_224

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:���������!2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:���������!2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:���������!:O K
'
_output_shapes
:���������!
 
_user_specified_nameinputs
�

�
#__inference_model_layer_call_fn_538

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*:
_output_shapes(
&:���������d:���������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU2*0J 8*G
fBR@
>__inference_model_layer_call_and_return_conditional_losses_3822
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������d2

Identity�

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*>
_input_shapes-
+:���������!::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������!
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
�
>__inference_model_layer_call_and_return_conditional_losses_316
input_1
	dense_297
	dense_299	
v_303	
v_305

pi_308

pi_310
identity

identity_1��dense/StatefulPartitionedCall�pi/StatefulPartitionedCall�v/StatefulPartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCallinput_1	dense_297	dense_299*
Tin
2*
Tout
2*'
_output_shapes
:���������!*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*G
fBR@
>__inference_dense_layer_call_and_return_conditional_losses_1912
dense/StatefulPartitionedCall�
dropout/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:���������!* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*I
fDRB
@__inference_dropout_layer_call_and_return_conditional_losses_2242
dropout/PartitionedCall�
v/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0v_303v_305*
Tin
2*
Tout
2*'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*C
f>R<
:__inference_v_layer_call_and_return_conditional_losses_2482
v/StatefulPartitionedCall�
pi/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0pi_308pi_310*
Tin
2*
Tout
2*'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*D
f?R=
;__inference_pi_layer_call_and_return_conditional_losses_2752
pi/StatefulPartitionedCall�
dense/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense/kernel/Regularizer/Const�
IdentityIdentity#pi/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall^pi/StatefulPartitionedCall^v/StatefulPartitionedCall*
T0*'
_output_shapes
:���������d2

Identity�

Identity_1Identity"v/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall^pi/StatefulPartitionedCall^v/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*>
_input_shapes-
+:���������!::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall28
pi/StatefulPartitionedCallpi/StatefulPartitionedCall26
v/StatefulPartitionedCallv/StatefulPartitionedCall:P L
'
_output_shapes
:���������!
!
_user_specified_name	input_1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�-
�
__inference__traced_restore_731
file_prefix!
assignvariableop_dense_kernel!
assignvariableop_1_dense_bias 
assignvariableop_2_pi_kernel
assignvariableop_3_pi_bias
assignvariableop_4_v_kernel
assignvariableop_5_v_bias
assignvariableop_6_decay$
 assignvariableop_7_learning_rate
assignvariableop_8_momentum
identity_10��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_2�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�	RestoreV2�RestoreV2_1�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*�
value�B�	B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE2
RestoreV2/tensor_names�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*%
valueB	B B B B B B B B B 2
RestoreV2/shape_and_slices�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*8
_output_shapes&
$:::::::::*
dtypes
2	2
	RestoreV2X
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:2

Identity�
AssignVariableOpAssignVariableOpassignvariableop_dense_kernelIdentity:output:0*
_output_shapes
 *
dtype02
AssignVariableOp\

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_biasIdentity_1:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1\

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOpassignvariableop_2_pi_kernelIdentity_2:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_2\

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOpassignvariableop_3_pi_biasIdentity_3:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_3\

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:2

Identity_4�
AssignVariableOp_4AssignVariableOpassignvariableop_4_v_kernelIdentity_4:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_4\

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:2

Identity_5�
AssignVariableOp_5AssignVariableOpassignvariableop_5_v_biasIdentity_5:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_5\

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:2

Identity_6�
AssignVariableOp_6AssignVariableOpassignvariableop_6_decayIdentity_6:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_6\

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:2

Identity_7�
AssignVariableOp_7AssignVariableOp assignvariableop_7_learning_rateIdentity_7:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_7\

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:2

Identity_8�
AssignVariableOp_8AssignVariableOpassignvariableop_8_momentumIdentity_8:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_8�
RestoreV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2_1/tensor_names�
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
RestoreV2_1/shape_and_slices�
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
_output_shapes
:*
dtypes
22
RestoreV2_19
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp�

Identity_9Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_9�
Identity_10IdentityIdentity_9:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: 2
Identity_10"#
identity_10Identity_10:output:0*9
_input_shapes(
&: :::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82
	RestoreV2	RestoreV22
RestoreV2_1RestoreV2_1:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :
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
: 
�
^
%__inference_dropout_layer_call_fn_587

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*'
_output_shapes
:���������!* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*I
fDRB
@__inference_dropout_layer_call_and_return_conditional_losses_2192
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������!2

Identity"
identityIdentity:output:0*&
_input_shapes
:���������!22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������!
 
_user_specified_nameinputs
�

�
#__inference_model_layer_call_fn_519

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*:
_output_shapes(
&:���������d:���������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU2*0J 8*G
fBR@
>__inference_model_layer_call_and_return_conditional_losses_3412
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������d2

Identity�

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*>
_input_shapes-
+:���������!::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������!
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�

�
!__inference_signature_wrapper_427
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*:
_output_shapes(
&:���������d:���������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU2*0J 8*'
f"R 
__inference__wrapped_model_1702
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������d2

Identity�

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*>
_input_shapes-
+:���������!::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:���������!
!
_user_specified_name	input_1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: "�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
;
input_10
serving_default_input_1:0���������!6
pi0
StatefulPartitionedCall:0���������d5
v0
StatefulPartitionedCall:1���������tensorflow/serving/predict:��
�(
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
	optimizer
loss
regularization_losses
	trainable_variables

	variables
	keras_api

signatures
?_default_save_signature
@__call__
*A&call_and_return_all_conditional_losses"�%
_tf_keras_model�%{"class_name": "Model", "name": "model", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 33]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 33, "activation": "swish", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0, "noise_shape": null, "seed": null}, "name": "dropout", "inbound_nodes": [[["dense", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "pi", "trainable": true, "dtype": "float32", "units": 100, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "pi", "inbound_nodes": [[["dropout", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "v", "trainable": true, "dtype": "float32", "units": 1, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "v", "inbound_nodes": [[["dropout", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["pi", 0, 0], ["v", 0, 0]]}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 33]}, "is_graph_network": true, "keras_version": "2.3.0-tf", "backend": "tensorflow", "model_config": {"class_name": "Model", "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 33]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 33, "activation": "swish", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0, "noise_shape": null, "seed": null}, "name": "dropout", "inbound_nodes": [[["dense", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "pi", "trainable": true, "dtype": "float32", "units": 100, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "pi", "inbound_nodes": [[["dropout", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "v", "trainable": true, "dtype": "float32", "units": 1, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "v", "inbound_nodes": [[["dropout", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["pi", 0, 0], ["v", 0, 0]]}}, "training_config": {"loss": ["categorical_crossentropy", "mean_squared_error"], "metrics": ["accuracy", "categorical_crossentropy", "mean_squared_error"], "weighted_metrics": null, "loss_weights": null, "sample_weight_mode": null, "optimizer_config": {"class_name": "SGD", "config": {"name": "SGD", "learning_rate": 0.009999999776482582, "decay": 0.0, "momentum": 0.0, "nesterov": false}}}}
�"�
_tf_keras_input_layer�{"class_name": "InputLayer", "name": "input_1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 33]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 33]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}
�

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
B__call__
*C&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 33, "activation": "swish", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 33}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 33]}}
�
regularization_losses
trainable_variables
	variables
	keras_api
D__call__
*E&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Dropout", "name": "dropout", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0, "noise_shape": null, "seed": null}}
�

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
F__call__
*G&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Dense", "name": "pi", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "pi", "trainable": true, "dtype": "float32", "units": 100, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 33}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 33]}}
�

kernel
bias
regularization_losses
 trainable_variables
!	variables
"	keras_api
H__call__
*I&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Dense", "name": "v", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "v", "trainable": true, "dtype": "float32", "units": 1, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 33}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 33]}}
?
	#decay
$learning_rate
%momentum"
	optimizer
 "
trackable_list_wrapper
'
J0"
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
�
&metrics
'non_trainable_variables

(layers
regularization_losses
)layer_regularization_losses
	trainable_variables

	variables
*layer_metrics
@__call__
?_default_save_signature
*A&call_and_return_all_conditional_losses
&A"call_and_return_conditional_losses"
_generic_user_object
,
Kserving_default"
signature_map
:!!2dense/kernel
:!2
dense/bias
'
J0"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
+metrics
,non_trainable_variables

-layers
regularization_losses
.layer_regularization_losses
trainable_variables
	variables
/layer_metrics
B__call__
*C&call_and_return_all_conditional_losses
&C"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
0metrics
1non_trainable_variables

2layers
regularization_losses
3layer_regularization_losses
trainable_variables
	variables
4layer_metrics
D__call__
*E&call_and_return_all_conditional_losses
&E"call_and_return_conditional_losses"
_generic_user_object
:!d2	pi/kernel
:d2pi/bias
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
�
5metrics
6non_trainable_variables

7layers
regularization_losses
8layer_regularization_losses
trainable_variables
	variables
9layer_metrics
F__call__
*G&call_and_return_all_conditional_losses
&G"call_and_return_conditional_losses"
_generic_user_object
:!2v/kernel
:2v/bias
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
�
:metrics
;non_trainable_variables

<layers
regularization_losses
=layer_regularization_losses
 trainable_variables
!	variables
>layer_metrics
H__call__
*I&call_and_return_all_conditional_losses
&I"call_and_return_conditional_losses"
_generic_user_object
: (2decay
: (2learning_rate
: (2momentum
 "
trackable_list_wrapper
 "
trackable_list_wrapper
C
0
1
2
3
4"
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
'
J0"
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
�2�
__inference__wrapped_model_170�
���
FullArgSpec
args� 
varargsjargs
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *&�#
!�
input_1���������!
�2�
#__inference_model_layer_call_fn_519
#__inference_model_layer_call_fn_358
#__inference_model_layer_call_fn_399
#__inference_model_layer_call_fn_538�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
>__inference_model_layer_call_and_return_conditional_losses_316
>__inference_model_layer_call_and_return_conditional_losses_294
>__inference_model_layer_call_and_return_conditional_losses_500
>__inference_model_layer_call_and_return_conditional_losses_467�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
#__inference_dense_layer_call_fn_565�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
>__inference_dense_layer_call_and_return_conditional_losses_556�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
%__inference_dropout_layer_call_fn_592
%__inference_dropout_layer_call_fn_587�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
@__inference_dropout_layer_call_and_return_conditional_losses_582
@__inference_dropout_layer_call_and_return_conditional_losses_577�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
 __inference_pi_layer_call_fn_612�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
;__inference_pi_layer_call_and_return_conditional_losses_603�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
__inference_v_layer_call_fn_632�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
:__inference_v_layer_call_and_return_conditional_losses_623�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
__inference_loss_fn_0_637�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
0B.
!__inference_signature_wrapper_427input_1�
__inference__wrapped_model_170�0�-
&�#
!�
input_1���������!
� "I�F
"
pi�
pi���������d
 
v�
v����������
>__inference_dense_layer_call_and_return_conditional_losses_556\/�,
%�"
 �
inputs���������!
� "%�"
�
0���������!
� v
#__inference_dense_layer_call_fn_565O/�,
%�"
 �
inputs���������!
� "����������!�
@__inference_dropout_layer_call_and_return_conditional_losses_577\3�0
)�&
 �
inputs���������!
p
� "%�"
�
0���������!
� �
@__inference_dropout_layer_call_and_return_conditional_losses_582\3�0
)�&
 �
inputs���������!
p 
� "%�"
�
0���������!
� x
%__inference_dropout_layer_call_fn_587O3�0
)�&
 �
inputs���������!
p
� "����������!x
%__inference_dropout_layer_call_fn_592O3�0
)�&
 �
inputs���������!
p 
� "����������!5
__inference_loss_fn_0_637�

� 
� "� �
>__inference_model_layer_call_and_return_conditional_losses_294�8�5
.�+
!�
input_1���������!
p

 
� "K�H
A�>
�
0/0���������d
�
0/1���������
� �
>__inference_model_layer_call_and_return_conditional_losses_316�8�5
.�+
!�
input_1���������!
p 

 
� "K�H
A�>
�
0/0���������d
�
0/1���������
� �
>__inference_model_layer_call_and_return_conditional_losses_467�7�4
-�*
 �
inputs���������!
p

 
� "K�H
A�>
�
0/0���������d
�
0/1���������
� �
>__inference_model_layer_call_and_return_conditional_losses_500�7�4
-�*
 �
inputs���������!
p 

 
� "K�H
A�>
�
0/0���������d
�
0/1���������
� �
#__inference_model_layer_call_fn_358�8�5
.�+
!�
input_1���������!
p

 
� "=�:
�
0���������d
�
1����������
#__inference_model_layer_call_fn_399�8�5
.�+
!�
input_1���������!
p 

 
� "=�:
�
0���������d
�
1����������
#__inference_model_layer_call_fn_519�7�4
-�*
 �
inputs���������!
p

 
� "=�:
�
0���������d
�
1����������
#__inference_model_layer_call_fn_538�7�4
-�*
 �
inputs���������!
p 

 
� "=�:
�
0���������d
�
1����������
;__inference_pi_layer_call_and_return_conditional_losses_603\/�,
%�"
 �
inputs���������!
� "%�"
�
0���������d
� s
 __inference_pi_layer_call_fn_612O/�,
%�"
 �
inputs���������!
� "����������d�
!__inference_signature_wrapper_427�;�8
� 
1�.
,
input_1!�
input_1���������!"I�F
"
pi�
pi���������d
 
v�
v����������
:__inference_v_layer_call_and_return_conditional_losses_623\/�,
%�"
 �
inputs���������!
� "%�"
�
0���������
� r
__inference_v_layer_call_fn_632O/�,
%�"
 �
inputs���������!
� "����������