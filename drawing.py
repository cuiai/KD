import numpy as np
import matplotlib.pyplot as plt
# a = []
# a.append((1,2))
# a.append((3,4))
# print(a)
# b = np.array(a)
# np.save('a.npy', b)
# teacher_data = np.load('teacher_data.npy')
# student_noKD_data = np.load('student_noKD_data.npy')
# student_KD_data = np.load('student_KD_data.npy')
# print(teacher_data)
epochs = 10

# plt.subplot(2, 1, 1)
# plt.title('Test accuracy')
# x = list(range(1, epochs+1))
# plt.plot(x, [teacher_data[i][1] for i in range(epochs)], label='teacher')
# plt.plot(x, [student_noKD_data[i][1] for i in range(epochs)], label='student_noKD_data')
# plt.plot(x, [student_KD_data[i][1] for i in range(epochs)], label='student_KD_data')
# plt.legend()
#
# plt.subplot(2, 1, 2)
# plt.title('Test loss')
# plt.plot(x, [teacher_data[i][0] for i in range(epochs)], label='teacher')
# plt.plot(x, [student_noKD_data[i][0] for i in range(epochs)], label='student_noKD_data')
# plt.plot(x, [student_KD_data[i][0] for i in range(epochs)], label='student_KD_data')
# plt.legend()
teacher_acc = np.load("draw_data/teacher_test_acc.npy")
teacher_loss = np.load("draw_data/teacher_test_loss.npy")
student_Kd_acc = np.load("draw_data/student_Kd_test_acc.npy")
student_Kd_loss = np.load("draw_data/student_noKd_test_loss.npy")
student_noKd_acc = np.load("draw_data/student_noKd_test_acc.npy")
student_noKd_loss = np.load("draw_data/student_Kd_test_loss.npy")

plt.subplot(2, 1, 1)
plt.title('Test accuracy')
x = list(range(1, epochs+1))
plt.plot(x, teacher_acc, label='teacher')
plt.plot(x, student_Kd_acc, label='student_noKD_data')
plt.plot(x, student_noKd_acc, label='student_KD_data')
plt.legend(loc=2)

plt.subplot(2, 1, 2)
plt.title('Test loss')
plt.plot(x, teacher_loss, label='teacher')
plt.plot(x, student_Kd_loss, label='student_noKD_data')
plt.plot(x, student_noKd_loss, label='student_KD_data')
plt.legend(loc=2)

plt.show()



