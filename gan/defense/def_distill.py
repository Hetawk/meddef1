import torch
import torch.nn.functional as F
import logging

class DefDistill:
    def __init__(self, teacher_model, student_model, temperature=10.0):
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.temperature = temperature

    def distillation_loss(self, student_outputs, teacher_outputs):
        """
        Compute the distillation loss between the student and teacher outputs.
        """
        teacher_outputs = F.softmax(teacher_outputs / self.temperature, dim=1)
        student_outputs = F.log_softmax(student_outputs / self.temperature, dim=1)
        loss = F.kl_div(student_outputs, teacher_outputs, reduction='batchmean') * (self.temperature ** 2)
        return loss

    def distill(self, inputs):
        """
        Perform distillation on the given inputs.
        """
        self.teacher_model.eval()
        self.student_model.eval()

        with torch.no_grad():
            teacher_outputs = self.teacher_model(inputs)

        student_outputs = self.student_model(inputs)
        loss = self.distillation_loss(student_outputs, teacher_outputs)

        logging.info(f'Distillation loss: {loss.item():.4f}')
        return loss