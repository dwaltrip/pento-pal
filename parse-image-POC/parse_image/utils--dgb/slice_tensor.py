
def slice_tensor(t, slice_specs):
    # Prepare a list of slices for each dimension
    slices = [slice(None)] * t.dim()  # Start with full slices

    # Replace the specific slices
    for spec in slice_specs:
        dim = spec['dim']
        start = spec.get('start')  # Will be None if not provided
        end = spec.get('end')  # Will be None if not provided
        slices[dim] = slice(start, end)

    # Apply the slices to the tensor
    sliced_t = t[slices]

    return sliced_t
