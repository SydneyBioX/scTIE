import torch
import torch.nn as nn
from copy import deepcopy


class VanillaBackprop:
    """
    Produces gradients generated with vanilla back propagation from the image
    """

    def __init__(self, model):
        self.model = model
        self.gradients = None
        # Put model in evaluation mode
        self.model.eval()
        # Hook the first layer to get the gradient

    def hook_input(self, input_tensor):
        def hook_function(grad_in):
            self.gradients = grad_in

        input_tensor.register_hook(hook_function)

    def generate_gradients(self, input_image, target_class):
        # Forward
        self.gradients = None

        input_image = deepcopy(input_image)
        input_image.requires_grad = True
        self.hook_input(input_image)
        # print(input_image.shape)
        model_output = self.model(input_image)

        # Zero grads
        self.model.zero_grad()
        # Target for backprop
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_().cuda()
        one_hot_output[0][target_class] = 1
        # print(one_hot_output.shape)
        # Backward pass
        model_output.backward(gradient=one_hot_output)
        # model_output.backward()
        # Convert Pytorch variable to numpy array
        # [0] to get rid of the first channel (1,3,224,224)
        # gradients_as_arr = self.gradients.data.cpu().numpy()[0]
        # print(input_image.grad.shape)
        gradients_as_arr = self.gradients.data.cpu().numpy()[0]
        # print(gradients_as_arr.shape)
        # print(input_image.grad.shape)
        return gradients_as_arr


class GuidedBackprop:
    """
    Produces gradients generated with guided back propagation from the given image
    """

    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.forward_relu_outputs = []
        # Put model in evaluation mode
        self.model.eval()
        self.update_relus()
        # self.hook_layers()

    def hook_input(self, input_tensor):
        def hook_function(grad_in):
            self.gradients = grad_in

        input_tensor.register_hook(hook_function)

    def update_relus(self):
        """
        Updates relu activation functions so that
            1- stores output in forward pass
            2- imputes zero for gradient values that are less than zero
        """

        def relu_backward_hook_function(module, grad_in, grad_out):
            """
            If there is a negative gradient, change it to zero
            """
            # Get last forward output
            corresponding_forward_output = self.forward_relu_outputs[-1]
            corresponding_forward_output[corresponding_forward_output > 0] = 1
            modified_grad_out = corresponding_forward_output * torch.clamp(grad_in[0], min=0.0)
            del self.forward_relu_outputs[-1]  # Remove last forward output
            return (modified_grad_out,)

        def relu_forward_hook_function(module, ten_in, ten_out):
            """
            Store results of forward pass
            """
            self.forward_relu_outputs.append(ten_out)

        # Loop through layers, hook up ReLU and LeakyReLUs
        for pos, module in self.model._modules.items():
            if isinstance(module, nn.ReLU):
                module.register_backward_hook(relu_backward_hook_function)
                module.register_forward_hook(relu_forward_hook_function)
            elif isinstance(module, nn.LeakyReLU):
                module.register_backward_hook(relu_backward_hook_function)
                module.register_forward_hook(relu_forward_hook_function)

    def generate_gradients(self, input_image, target_class):
        # Forward pass
        input_image = deepcopy(input_image)
        input_image.requires_grad = True
        self.hook_input(input_image)
        model_output = self.model(input_image)
        # Zero gradients
        self.model.zero_grad()
        # Target for backprop
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_().cuda()
        one_hot_output[0][target_class] = 1
        # Backward pass
        model_output.backward(gradient=one_hot_output)
        # Convert Pytorch variable to numpy array
        # [0] to get rid of the first channel (1,3,224,224)
        gradients_as_arr = self.gradients.data.cpu().numpy()[0]
        return gradients_as_arr
