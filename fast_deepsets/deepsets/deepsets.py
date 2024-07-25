# Implementation of the permutation invariant Deep Sets network from the
# https://arxiv.org/abs/1703.06114 paper.

import numpy as np

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.layers as KL
import qkeras

from fast_deepsets.util import flops



class DeepSetsInv(keras.Model):
    """Deep sets permutation invariant graph network https://arxiv.org/abs/1703.06114.

    Attributes:
        input_size: Tuple with the shape of the input data.
        phi_layers: List of number of nodes for each layer of the phi network.
        rho_layers: List of number of nodes for each layer of the rho network.
        activ: String that specifies Activation function to use between the dense layers.
        aggreg: String that specifies the type of aggregator to use after the phi net.
        output_dim: The output dimension of the network. For a supervised task, this is
            equal to the number of classes.
    """
    def __init__(
        self,
        input_size: tuple,
        phi_layers: list = [32, 32, 32],
        rho_layers: list = [16],
        output_dim: int = 5,
        activ: str = "relu",
        aggreg: str =  "mean"
    ):
        super(DeepSetsInv, self).__init__(name="InvariantDeepsets")
        self.input_size = input_size
        self.output_dim = output_dim
        self.phi_layers = phi_layers
        self.rho_layers = rho_layers
        self.aggreg = aggreg
        self.activ = activ
        self.flops = {"layer": 0, "activation": 0, "bottleneck": 0}

        self._build_phi()
        self._build_agg()
        self._build_rho()
        self.output_layer = KL.Dense(self.output_dim, name="OutputLayer")

    def _build_phi(self):
        input_shape = list(self.input_size[1:])
        self.phi = keras.Sequential(name="PhiNetwork")
        for layer in self.phi_layers:
            self.phi.add(KL.Dense(layer))
            self.flops["layer"] += flops.get_flops_dense(input_shape, layer)
            input_shape[-1] = layer

            self.phi.add(KL.Activation(self.activ))
            self.flops["activation"] += flops.get_flops_activ(input_shape, self.activ)

    def _build_agg(self):
        switcher = {
            "mean": lambda: self._get_mean_aggregator(),
            "max": lambda: self._get_max_aggregator(),
        }
        self.agg = switcher.get(self.aggreg, lambda: None)()
        if self.agg is None:
            raise ValueError(
                "Given aggregation string is not implemented. "
                "See deepsets.py and add string and corresponding object there."
            )

    def _build_rho(self):
        input_shape = self.phi_layers[-1]
        self.rho = keras.Sequential(name="RhoNetwork")
        for layer in self.rho_layers:
            self.rho.add(KL.Dense(layer))
            self.flops["layer"] += flops.get_flops_dense(input_shape, layer)
            input_shape = layer

            self.rho.add(KL.Activation(self.activ))
            self.flops["activation"] += flops.get_flops_activ(input_shape, self.activ)

        # Get the flops of the output layer.
        self.flops["layer"] += flops.get_flops_dense(layer, self.output_dim)

    def _get_mean_aggregator(self):
        """Get mean aggregator object and calculate number of flops."""
        # Sum number of inputs into the aggregator + 1 division times number of feats.
        self.flops["bottleneck"] = (self.phi_layers[-1] + 1)*self.input_size[-1]

        return tf.reduce_mean

    def _get_max_aggregator(self):
        """Get max aggregator object and calculate number of flops."""
        # FLOPs calculation WIP.

        return tf.reduce_max

    def save_flops(self, outdir):
        """Saves flops of model to output directory."""
        flops_file_path = os.path.join(outdir, "flops.json")
        with open(flops_file_path, "w") as file:
            json.dump(flops, file)
        print(tcols.OKGREEN + "Saved flops information to json file." + tcols.ENDC)

    def call(self, inputs: np.ndarray, **kwargs):
        phi_output = self.phi(inputs)
        agg_output = self.agg(phi_output, axis=1)
        rho_output = self.rho(agg_output)
        logits = self.output_layer(rho_output)

        return logits

# ----------------------------------

class DeepSetsInv_custom(keras.Model):
    def __init__(
        self,
        input_size: tuple,
        phi_layers: list = [8, 8],
        phi_activations: list = ["leaky_relu", "relu"],
        phi_normalizations: list = ["batch", None],
        rho_layers: list = [16, 16],
        rho_activations: list = ["leaky_relu", "relu", "leaky_relu"],
        rho_normalizations: list = ["batch", "batch", None],
        output_dim: int = 5,
        activ: str = "relu",
        leaky_relu_alpha: float = 0.01,
        aggreg: str = "mean"
    ):
        super(DeepSetsInv_custom, self).__init__(name="customInvariantDeepsets")
        self.input_size = input_size
        self.output_dim = output_dim
        self.phi_layers = phi_layers
        self.phi_activations = phi_activations
        self.phi_normalizations = phi_normalizations
        self.rho_layers = rho_layers
        self.rho_activations = rho_activations
        self.rho_normalizations = rho_normalizations
        self.aggreg = aggreg
        self.activ = activ
        self.leaky_relu_alpha = leaky_relu_alpha
        self.flops = {"layer": 0, "activation": 0, "bottleneck": 0}

        self._build_phi()
        self._build_agg()
        self._build_rho()
        self.output_layer = KL.Dense(self.output_dim, name="OutputLayer")

    def _get_activation(self, activation_name):
        if activation_name == "relu":
            return KL.ReLU()
        elif activation_name == "leaky_relu":
            return KL.LeakyReLU(alpha=self.leaky_relu_alpha)
        else:
            return KL.Activation(activation_name)


    def _build_phi(self):
        self.phi = keras.Sequential(name="PhiNetwork")
        for layer, activation, normalization in zip(self.phi_layers, self.phi_activations, self.phi_normalizations):
            self.phi.add(KL.Dense(layer))
            if normalization == "batch":
                self.phi.add(KL.BatchNormalization(axis=2))
            self.phi.add(self._get_activation(activation))


    def _build_agg(self):
        switcher = {
            "mean": lambda: self._get_mean_aggregator(),
            "max": lambda: self._get_max_aggregator(),
        }
        self.agg = switcher.get(self.aggreg, lambda: None)()
        if self.agg is None:
            raise ValueError(
                "Given aggregation string is not implemented. "
                "See deepsets.py and add string and corresponding object there."
            )

    def _get_mean_aggregator(self):
        """Get mean aggregator object and calculate number of flops."""
        # Sum number of inputs into the aggregator + 1 division times number of feats.
        self.flops["bottleneck"] = (self.phi_layers[-1] + 1)*self.input_size[-1]

        return tf.reduce_mean

    def _get_max_aggregator(self):
        """Get max aggregator object and calculate number of flops."""
        # FLOPs calculation WIP.

        return tf.reduce_max

    def _build_rho(self):
        self.rho = keras.Sequential(name="RhoNetwork")
        for layer, activation, normalization in zip(self.rho_layers, self.rho_activations, self.rho_normalizations):
            self.rho.add(KL.Dense(layer))
            if normalization == "batch":
                self.rho.add(KL.BatchNormalization(axis=1))
            self.rho.add(self._get_activation(activation))


    def _get_flops_dense(self, input_dim, output_dim):
        return input_dim * output_dim * 2

    def _get_flops_activ(self, dim):
        return dim

    def call(self, inputs: np.ndarray, **kwargs):
        phi_output = self.phi(inputs)
        agg_output = self.agg(phi_output, axis=1)
        rho_output = self.rho(agg_output)
        logits = self.output_layer(rho_output)

        return logits

    def get_config(self):
        config = super().get_config()
        config.update({
            "input_size": self.input_size,
            "phi_layers": self.phi_layers,
            "phi_activations": self.phi_activations,
            "phi_normalizations": self.phi_normalizations,
            "rho_layers": self.rho_layers,
            "rho_activations": self.rho_activations,
            "rho_normalizations": self.rho_normalizations,
            "output_dim": self.output_dim,
            "activ": self.activ,
            "leaky_relu_alpha": self.leaky_relu_alpha,
            "aggreg": self.aggreg,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

