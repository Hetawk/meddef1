import torch
import torch.nn.functional as F
import logging

class Randomization:
    def __init__(self, model, train_loader, test_loader, randomization_factor=0.1):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.randomization_factor = randomization_factor
        logging.info("Randomization initialized.")

    def train(self, epochs):
        self.model.train()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        for epoch in range(epochs):
            for data, target in self.train_loader:
                data.requires_grad = True
                output = self.model(data)
                loss = F.cross_entropy(output, target)
                self.model.zero_grad()
                loss.backward()
                # Apply randomization technique here
                data = data + self.randomization_factor * torch.randn_like(data)
                adv_output = self.model(data)
                adv_loss = F.cross_entropy(adv_output, target)
                total_loss = loss + adv_loss
                total_loss.backward()
                optimizer.step()
            logging.info(f'Epoch {epoch+1}/{epochs} completed.')

    def test(self):
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                output = self.model(data)
                test_loss += F.cross_entropy(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(self.test_loader.dataset)
        accuracy = 100. * correct / len(self.test_loader.dataset)
        logging.info(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(self.test_loader.dataset)} ({accuracy:.0f}%)')
