# Deepsets models in a format friendly for synthetisation. For more details on the
# architecture see the deepsets.py file.

import numpy as np

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.layers as KL
import qkeras


def deepsets_invariant_synth(
    input_size: tuple,
    phi_layers: list = [32, 32, 32],
    rho_layers: list = [16],
    output_dim: int = 5,
    activ: str = "relu",
    aggreg: str =  "mean",
    aggreg_precision: dict = {"bits": 20, "integer": 10},
    nbits: int = 8
):
    """Deep sets permutation invariant graph network https://arxiv.org/abs/1703.06114.

    The weights of this model are quantised to a given number of bits.

    Attributes:
        input_size: Tuple with the shape of the input data.
        phi_layers: List of number of nodes for each layer of the phi network.
        rho_layers: List of number of nodes for each layer of the rho network.
        activ: String that specifies Activation function to use between the dense layers.
        aggreg: String that specifies the type of aggregator to use after the phi net.
        output_dim: The output dimension of the network. For a supervised task, this is
            equal to the number of classes.
        nbits: Number of bits to quantise the weights of the model to.
    """
    quant = format_quantiser(nbits)
    # Activation precision is fixed to 8. Decreasing this precision would have severe
    # impacts on the performance of the model.
    activ = format_qactivation(activ, 8)

    deepsets_input = keras.Input(shape=input_size[1:], name="input_layer")

    # Phi network.
    x = qkeras.QDense(
            phi_layers[0], kernel_quantizer=quant, bias_quantizer=quant, name=f"phi{1}"
        )(deepsets_input)
    x = qkeras.QActivation(activ)(x)
    for i, layer in enumerate(phi_layers[1:]):
        x = qkeras.QDense(
                layer, kernel_quantizer=quant, bias_quantizer=quant, name=f"phi{i+2}"
            )(x)
        x = qkeras.QActivation(activ)(x)

    # Trick to change the precision of the input to the aggregator.
    x = qkeras.QActivation(
            qkeras.quantized_bits(**aggreg_precision, symmetric=0, keep_negative=1)
        )(x)

    # Aggregator
    agg = choose_aggregator(aggreg)
    x = agg(x)

    # Rho network.
    for i, layer in enumerate(rho_layers):
        x = qkeras.QDense(
                layer, kernel_quantizer=quant, bias_quantizer=quant, name=f"rho{i+1}"
            )(x)
        x = qkeras.QActivation(activ)(x)

    deepsets_output = KL.Dense(output_dim)(x)
    deepsets_output = KL.Softmax()(deepsets_output)
    deepsets = keras.Model(deepsets_input, deepsets_output, name="deepsets_invariant")

    return deepsets


def choose_aggregator(choice: str):
    """Choose the aggregator keras object based on an input string."""
    switcher = {
        "mean": lambda: KL.GlobalAveragePooling1D(),
        "max": lambda: KL.GlobalMaxPooling1D(),
    }
    agg = switcher.get(choice, lambda: None)()
    if agg is None:
        raise ValueError(
            "Given aggregation string is not implemented. "
            "See deepsets.py and add string and corresponding object there."
        )

    return agg


def format_quantiser(nbits: int):
    """Format the quantisation of the ml floats in a QKeras way."""
    if nbits == 1:
        return "binary(alpha=1)"
    elif nbits == 2:
        return "ternary(alpha=1)"
    else:
        return f"quantized_bits({nbits}, 0, alpha=1)"


def format_qactivation(activation: str, nbits: int) -> str:
    """Format the activation function strings in a QKeras friendly way."""
    return f"quantized_{activation}({nbits}, 0)"




def deepsets_invariant_synth_custom2(
    input_size: tuple,
    phi_layers: list = [32, 32, 32],
    phi_activations: list = ["relu", "relu", "relu"],
    phi_normalizations: list = [None, None, None],
    rho_layers: list = [16],
    rho_activations: list = ["relu"],
    rho_normalizations: list = [None],
    output_dim: int = 5,
    aggreg: str = "mean",
    aggreg_precision: dict = {"bits": 20, "integer": 10},
    nbits: int = 8,
    leaky_relu_alpha: float = 0.01
):
    """Deep sets permutation invariant graph network with custom activations and batch normalization.

    Attributes:
        input_size: Tuple with the shape of the input data.
        phi_layers: List of number of nodes for each layer of the phi network.
        phi_activations: List of activation functions for each layer of the phi network.
        phi_normalizations: List of normalization types for each layer of the phi network.
        rho_layers: List of number of nodes for each layer of the rho network.
        rho_activations: List of activation functions for each layer of the rho network.
        rho_normalizations: List of normalization types for each layer of the rho network.
        aggreg: String that specifies the type of aggregator to use after the phi net.
        output_dim: The output dimension of the network.
        aggreg_precision: Dictionary specifying the precision for the aggregator.
        nbits: Number of bits to quantise the weights of the model to.
        leaky_relu_alpha: Alpha value for LeakyReLU activation.
    """
    
    
    quant = format_quantiser(nbits)
    
    deepsets_input = keras.Input(shape=input_size[1:], name="input_layer")
    
    # Phi network
    x = deepsets_input
    # print(len(phi_layers))
    # print(len(phi_activations))
    # print(len(phi_normalizations))

    # for i in range(len(phi_activations)):



    for i, (layer, activation, normalization) in enumerate(zip(phi_layers, phi_activations, phi_normalizations)):
        print(f'layer: {layer}, activation: {activation}, normalization {normalization}')
        x = qkeras.QDense(layer, kernel_quantizer=quant, bias_quantizer=quant, name=f"phi{i+1}")(x)
        if normalization == "batch":
            print("batch norm set up")
            x = qkeras.QBatchNormalization(name=f"phi_bn{i+1}")(x)
        x = apply_activation(x, activation, nbits, leaky_relu_alpha, name=f"phi_activ{i+1}")
    
    # assert(1==0)

    # Aggregator
    x = qkeras.QActivation(
        qkeras.quantized_bits(**aggreg_precision, symmetric=0, keep_negative=1),
        name="pre_aggreg_quant"
    )(x)
    agg = choose_aggregator(aggreg)
    x = agg(x)
    
    # Rho network
    for i, (layer, activation, normalization) in enumerate(zip(rho_layers, rho_activations, rho_normalizations)):
        x = qkeras.QDense(layer, kernel_quantizer=quant, bias_quantizer=quant, name=f"rho{i+1}")(x)
        if normalization == "batch":
            x = qkeras.QBatchNormalization(name=f"rho_bn{i+1}")(x)
        x = apply_activation(x, activation, nbits, leaky_relu_alpha, name=f"rho_activ{i+1}")

    deepsets_output = KL.Dense(output_dim, name="output_dense")(x)
    if rho_normalizations[-1] == "batch":
        deepsets_output = qkeras.QBatchNormalization(name=f"output_bn")(deepsets_output)
    deepsets_output = apply_activation(deepsets_output, rho_activations[-1], nbits, leaky_relu_alpha, name="output_activ")

    deepsets_output = KL.Softmax(name="output_softmax")(deepsets_output)
    deepsets = keras.Model(deepsets_input, deepsets_output, name="deepsets_invariant")
    
    return deepsets

def choose_aggregator(choice: str):
    """Choose the aggregator keras object based on an input string."""
    switcher = {
        "mean": lambda: KL.GlobalAveragePooling1D(),
        "max": lambda: KL.GlobalMaxPooling1D(),
    }
    agg = switcher.get(choice, lambda: None)()
    if agg is None:
        raise ValueError(
            "Given aggregation string is not implemented. "
            "See deepsets.py and add string and corresponding object there."
        )
    return agg

def format_quantiser(nbits: int):
    """Format the quantisation of the ml floats in a QKeras way."""
    if nbits == 1:
        return "binary(alpha=1)"
    elif nbits == 2:
        return "ternary(alpha=1)"
    else:
        return f"quantized_bits({nbits}, 0, alpha=1)"

def apply_activation(x, activation: str, nbits: int, leaky_relu_alpha: float, name: str = None):
    """Apply the specified activation function."""
    if activation == "relu":
        return qkeras.QActivation(f"quantized_relu({nbits}, 0)", name=name)(x)
    elif activation == "leaky_relu":
        return KL.LeakyReLU(alpha=leaky_relu_alpha, name=name)(x)
    elif activation == "tanh":
        return qkeras.QActivation(f"quantized_tanh({nbits}, 0)", name=name)(x)
    elif activation == "sigmoid":
        return qkeras.QActivation(f"quantized_sigmoid({nbits}, 0)", name=name)(x)
    else:
        raise ValueError(f"Unsupported activation function: {activation}")

