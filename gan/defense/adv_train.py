import torch

class AdversarialTraining:
    def __init__(self, model, criterion, optimizer, epochs):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.epochs = epochs

    def defend(self, data, targets):
        data, targets = data.to(self.model.device), targets.to(self.model.device)

        for epoch in range(self.epochs):
            self.model.train()
            self.optimizer.zero_grad()

            outputs = self.model(data)
            loss = self.criterion(outputs, targets)

            loss.backward()
            self.optimizer.step()

        return data