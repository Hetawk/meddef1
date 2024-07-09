import torch
import torch.nn.functional as F
import logging

class EnsembleAdvTrain:
    def __init__(self, models, train_loader, test_loader, epsilon=0.3, alpha=0.01, iterations=40):
        self.models = models
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.epsilon = epsilon
        self.alpha = alpha
        self.iterations = iterations
        logging.info("EnsembleAdvTrain initialized.")

    def train(self, epochs):
        for model in self.models:
            model.train()
        optimizer = torch.optim.SGD(sum([list(model.parameters()) for model in self.models], []), lr=0.01, momentum=0.9)
        for epoch in range(epochs):
            for data, target in self.train_loader:
                data.requires_grad = True
                losses = []
                for model in self.models:
                    output = model(data)
                    loss = F.cross_entropy(output, target)
                    model.zero_grad()
                    loss.backward()
                    losses.append(loss)
                total_loss = sum(losses) / len(self.models)
                total_loss.backward()
                optimizer.step()
            logging.info(f'Epoch {epoch+1}/{epochs} completed.')

    def test(self):
        for model in self.models:
            model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                outputs = [model(data) for model in self.models]
                avg_output = sum(outputs) / len(self.models)
                test_loss += F.cross_entropy(avg_output, target, reduction='sum').item()
                pred = avg_output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(self.test_loader.dataset)
        accuracy = 100. * correct / len(self.test_loader.dataset)
        logging.info(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(self.test_loader.dataset)} ({accuracy:.0f}%)')
