import torch
from torch.autograd import Function
from torch.autograd import Variable

class EBLinear(Function):
    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, weight, bias=None):
        ctx.save_for_backward(input, weight, bias)
        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_variables

        ### start EB-SPECIFIC CODE  ###
        # Grad output is gradient of output

        # print("this is a {} linear layer ({})"
        #       .format('pos' if torch.use_pos_weights else 'neg', grad_output.sum().data[0]))

        # Enforce that weights are non-negative
        weight = weight.clamp(min=0) if torch.use_pos_weights else weight.clamp(max=0).abs()

        input.data = input.data - input.data.min() if input.data.min() < 0 else input.data
        grad_output /= input.mm(weight.t()).abs() + 1e-10 # normalize
        ### stop EB-SPECIFIC CODE  ###

        grad_input = grad_weight = grad_bias = None

        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
            ### start EB-SPECIFIC CODE  ###
            grad_input *= input
            ### stop EB-SPECIFIC CODE  ###

        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)


        return grad_input, grad_weight, grad_bias