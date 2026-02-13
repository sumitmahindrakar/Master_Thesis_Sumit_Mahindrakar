import torch

def node_accuracy(node_out, node_y, accuracy_threshold):

    device=node_out.device

    condition = torch.abs(node_y)> accuracy_threshold

    ones = torch.ones(node_y.shape).to(device)[condition]
    zeros = torch.zeros(node_y.shape).to(device)[condition]

    absolute_error = torch.abs(node_y[condition] - node_out[condition])
    relative_error = torch.div(absolute_error, torch.abs(node_y[condition]))
    relative_accuracy = torch.max(ones - relative_error, zeros)

    return relative_accuracy.sum(), torch.numel(relative_accuracy)


def calculate_accuracy_percentage(node_out, node_y, accuracy_threshold):

    correct, total = node_accuracy(node_out, node_y, accuracy_threshold)

    if total ==0:
        return 0.0
    
    return (correct / total).item() * 100


def detailed_accuracy(node_out, node_y, accuracy_threshold):

    results = {}

    output_ranges = {
        'displacement':(0,2),
        'moment_Y':(2,8),
        'moment_Z':(8,14),
        'shear_Y':(14,20),
        'shear_Z':(20,26),

    }

    for name, (start, end) in output_ranges.items():
        correct, total = node_accuracy(
            node_out[:,start:end],
            node_y[:, start:end],
            accuracy_threshold
        )
        if total > 0:
            results[name] = (correct / total).item()
        else:
            results[name] = 0.0

        correct, total = node_accuracy(node_out, node_y, accuracy_threshold)
        if total > 0:
            results['overall'] = (correct / total).item()
        else:
            results['overall'] = 0.0

        return results
