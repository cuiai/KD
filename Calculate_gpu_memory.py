from model import StudentNet, TeacherNet
from torchstat import stat
import torchvision.models as models
model = StudentNet()
stat(model, (1, 28, 28))