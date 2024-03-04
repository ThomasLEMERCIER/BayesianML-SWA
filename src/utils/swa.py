
def update_swa(swa_model, model, n):
    for swa_param, param in zip(swa_model.parameters(), model.parameters()):
        swa_param.data = (swa_param.data * n + param.data.detach()) / (n + 1)
