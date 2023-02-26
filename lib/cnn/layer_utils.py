from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


class sequential(object):
    def __init__(self, *args):
        """
        Sequential Object to serialize the NN layers
        Please read this code block and understand how it works
        """
        self.params = {}
        self.grads = {}
        self.layers = []
        self.paramName2Indices = {}
        self.layer_names = {}

        # process the parameters layer by layer
        for layer_cnt, layer in enumerate(args):
            for n, v in layer.params.items():
                self.params[n] = v
                self.paramName2Indices[n] = layer_cnt
            for n, v in layer.grads.items():
                self.grads[n] = v
            if layer.name in self.layer_names:
                raise ValueError("Existing name {}!".format(layer.name))
            self.layer_names[layer.name] = True
            self.layers.append(layer)

    def assign(self, name, val):
        # load the given values to the layer by name
        layer_cnt = self.paramName2Indices[name]
        self.layers[layer_cnt].params[name] = val

    def assign_grads(self, name, val):
        # load the given values to the layer by name
        layer_cnt = self.paramName2Indices[name]
        self.layers[layer_cnt].grads[name] = val

    def get_params(self, name):
        # return the parameters by name
        return self.params[name]

    def get_grads(self, name):
        # return the gradients by name
        return self.grads[name]

    def gather_params(self):
        """
        Collect the parameters of every submodules
        """
        for layer in self.layers:
            for n, v in layer.params.items():
                self.params[n] = v

    def gather_grads(self):
        """
        Collect the gradients of every submodules
        """
        for layer in self.layers:
            for n, v in layer.grads.items():
                self.grads[n] = v

    def apply_l1_regularization(self, lam):
        """
        Gather gradients for L1 regularization to every submodule
        """
        for layer in self.layers:
            for n, v in layer.grads.items():
                param = self.params[n]
                grad = (param > 0).astype(np.float32) - (param < 0).astype(np.float32)
                self.grads[n] += lam * grad

    def apply_l2_regularization(self, lam):
        """
        Gather gradients for L2 regularization to every submodule
        """
        for layer in self.layers:
            for n, v in layer.grads.items():
                self.grads[n] += lam * self.params[n]


    def load(self, pretrained):
        """
        Load a pretrained model by names
        """
        for layer in self.layers:
            if not hasattr(layer, "params"):
                continue
            for n, v in layer.params.items():
                if n in pretrained.keys():
                    layer.params[n] = pretrained[n].copy()
                    print ("Loading Params: {} Shape: {}".format(n, layer.params[n].shape))

class ConvLayer2D(object):
    def __init__(self, input_channels, kernel_size, number_filters, 
                stride=1, padding=0, init_scale=.02, name="conv"):
        
        self.name = name
        self.w_name = name + "_w"
        self.b_name = name + "_b"

        self.input_channels = input_channels
        self.kernel_size = kernel_size
        self.number_filters = number_filters
        self.stride = stride
        self.padding = padding

        self.params = {}
        self.grads = {}
        self.params[self.w_name] = init_scale * np.random.randn(kernel_size, kernel_size, 
                                                                input_channels, number_filters)
        self.params[self.b_name] = np.zeros(number_filters)
        self.grads[self.w_name] = None
        self.grads[self.b_name] = None
        self.meta = None
    
    def get_output_size(self, input_size):
        '''
        :param input_size - 4-D shape of input image tensor (batch_size, in_height, in_width, in_channels)
        :output a 4-D shape of the output after passing through this layer (batch_size, out_height, out_width, out_channels)
        '''
        output_shape = [None, None, None, None]
        #############################################################################
        # TODO: Implement the calculation to find the output size given the         #
        # parameters of this convolutional layer.                                   #
        #############################################################################
        input_height, input_width = input_size[1], input_size[2]
        kernel_height, kernel_width = self.kernel_size, self.kernel_size
        stride_h, stride_w = self.stride, self.stride
        pad_h, pad_w = self.padding, self.padding

        output_height = (input_height + 2 * pad_h - kernel_height) // stride_h + 1
        output_width = (input_width + 2 * pad_w - kernel_width) // stride_w + 1
        output_shape = (input_size[0], output_height, output_width, self.number_filters)
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return output_shape

    def forward(self, img):
        output = None
        assert len(img.shape) == 4, "expected batch of images, but received shape {}".format(img.shape)

        output_shape = self.get_output_size(img.shape)
        _ , input_height, input_width, _ = img.shape
        _, output_height, output_width, _ = output_shape

        #############################################################################
        # TODO: Implement the forward pass of a single convolutional layer.       #
        # Store the results in the variable "output" provided above.                #
        #############################################################################
        # Add padding if needed
        if self.padding > 0:
            img = np.pad(img, ((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)), mode='constant')

        # Compute the convolution using a nested loop over the batch and output channels
        output = np.zeros(output_shape)
        w = self.params[self.w_name]
        b = self.params[self.b_name]
        for n in range(img.shape[0]):  # iterate over batch size
            for k in range(self.number_filters):  # iterate over output channels
                for i in range(output_height):  # iterate over output height
                    for j in range(output_width):  # iterate over output width
                        # Extract the receptive field from the input image
                        h_start, h_end = i * self.stride, i * self.stride + self.kernel_size
                        w_start, w_end = j * self.stride, j * self.stride + self.kernel_size
                        receptive_field = img[n, h_start:h_end, w_start:w_end, :]
                        
                        # Compute the dot product with the filter weights and add the bias term
                        output[n, i, j, k] = np.sum(receptive_field * w[:, :, :, k]) + b[k]
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        self.meta = img
        
        return output


    def backward(self, dprev):
        img = self.meta
        if img is None:
            raise ValueError("No forward function called before for this module!")

        dimg, self.grads[self.w_name], self.grads[self.b_name] = None, None, None
        
        #############################################################################
        # TODO: Implement the backward pass of a single convolutional layer.        #
        # Store the computed gradients wrt weights and biases in self.grads with    #
        # corresponding name.                                                       #
        # Store the output gradients in the variable dimg provided above.           #
        #############################################################################
        d_w = np.zeros(self.params[self.w_name].shape)
        d_img = np.zeros(img.shape)
        it = np.nditer(dprev[0, :, :, 0], flags=["multi_index"], op_flags=["readwrite"])
        while not it.finished:
            i, j = it.multi_index
            i_stride = i*self.stride
            j_stride = j*self.stride
            mid_d = np.einsum("bhwd,bf->fhwd", img[:,i_stride:i_stride+self.kernel_size, j_stride:j_stride+self.kernel_size], dprev[:, i, j])
            d_img[:, i_stride:i_stride+self.kernel_size, j_stride:j_stride+self.kernel_size, :] += np.einsum("bf,hwdf->bhwd", dprev[:,i,j,:], self.params[self.w_name])
            
            d_w += np.einsum("fhwd->hwdf", mid_d)
            it.iternext()
            
        self.grads[self.w_name] = d_w
        self.grads[self.b_name] = np.einsum("bhwf->f", dprev)
        if self.padding:
            dimg = d_img[:, self.padding:-self.padding, self.padding:-self.padding, :]
    
                    

        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################

        self.meta = None
        return dimg


class MaxPoolingLayer(object):
    def __init__(self, pool_size, stride, name):
        self.name = name
        self.pool_size = pool_size
        self.stride = stride
        self.params = {}
        self.grads = {}
        self.meta = None

    def forward(self, img):
        output = None
        assert len(img.shape) == 4, "expected batch of images, but received shape {}".format(img.shape)
        
        #############################################################################
        # TODO: Implement the forward pass of a single maxpooling layer.            #
        # Store your results in the variable "output" provided above.               #
        #############################################################################
        N, H, W, C = img.shape
        HH, WW = self.pool_size, self.pool_size
        stride = self.stride

        H_out = int(1 + (H - HH) / stride)
        W_out = int(1 + (W - WW) / stride)

        out = np.zeros((N, H_out, W_out, C))

        for i in range(H_out):
            for j in range(W_out):
                out[:, i, j, :] = np.max(img[:, i*stride:i*stride+HH, j*stride:j*stride+WW, :], axis=(1,2))

        output = out
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        self.meta = img

        return output

    def backward(self, dprev):
        img = self.meta

        dimg = np.zeros_like(img)
        _, h_out, w_out, _ = dprev.shape
        h_pool, w_pool = self.pool_size,self.pool_size

        #############################################################################
        # TODO: Implement the backward pass of a single maxpool layer.              #
        # Store the computed gradients in self.grads with corresponding name.       #
        # Store the output gradients in the variable dimg provided above.           #
        #############################################################################
        for i in range(h_out):
            for j in range(w_out):
                h_start = i * self.stride
                h_end = h_start + h_pool
                w_start = j * self.stride
                w_end = w_start + w_pool
                window = img[:, h_start:h_end, w_start:w_end, :]
                max_val = np.max(window, axis=(1, 2))
                max_val = np.expand_dims(max_val, axis=(1, 2))
                mask = (window == max_val)
                dimg[:, h_start:h_end, w_start:w_end, :] += mask * np.expand_dims(np.expand_dims(dprev[:, i, j, :], axis=1), axis=1)
        


        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return dimg
