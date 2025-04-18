def replace_layers(model, old, new):
    for n, module in model.named_children():
        if len(list(module.children())) > 0:
            ## compound module, go inside it
            replace_layers(module, old, new)
        if isinstance(module, old):
            ## simple module
            try:
                n = int(n)
                model[n] = new(len(module.weight))
            except:
                setattr(model, n, new(len(module.weight)))


