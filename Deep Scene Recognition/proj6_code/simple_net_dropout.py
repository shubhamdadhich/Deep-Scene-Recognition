import torch
import torch.nn as nn


class SimpleNetDropout(nn.Module):
  def __init__(self):
    '''
    Init function to define the layers and loss function

    Note: Use 'sum' reduction in the loss_criterion. Read Pytorch documention
    to understand what it means
    '''
    super().__init__()

    self.cnn_layers = nn.Sequential()
    self.fc_layers = nn.Sequential()
    self.loss_criterion = None

    ###########################################################################
    # Student code begin
    ###########################################################################

    self.cnn_layers = nn.Sequential(

      nn.Conv2d(1, 10, kernel_size=5, stride=1),
      nn.ReLU(),
      nn.MaxPool2d(3, 3),

      nn.Conv2d(10, 20, kernel_size=5, stride=1),
      nn.ReLU(),
      nn.MaxPool2d(3, 3),
    )

    self.fc_layers = nn.Sequential(
      nn.Dropout(p=0.25),

      nn.Linear(500, 120),
      nn.ReLU(),

      # nn.Dropout(p=0.25),
      nn.Linear(120, 84),
      nn.ReLU(),

      # nn.Dropout(p=0.25),
      nn.Linear(84, 15),
    )

    self.loss_criterion = nn.CrossEntropyLoss(reduction='sum')

    ###########################################################################
    # Student code end
    ###########################################################################

  def forward(self, x: torch.tensor) -> torch.tensor:
    '''
    Perform the forward pass with the net

    Args:
    -   x: the input image [Dim: (N,C,H,W)]
    Returns:
    -   y: the output (raw scores) of the net [Dim: (N,15)]
    '''
    model_output = None
    ###########################################################################
    # Student code begin
    ###########################################################################

    y = self.cnn_layers(x)
    y = y.view(-1, 500)
    model_output = self.fc_layers(y)

    ###########################################################################
    # Student code end
    ###########################################################################
    return model_output
