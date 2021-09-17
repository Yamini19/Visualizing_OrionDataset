import re
import numpy as np
import matplotlib.pyplot as plt


def linearTransformation(Dataset):  # Function for Linear Transformation (Rescale 0-255)

    LinearTrans = np.zeros(shape=(len(Dataset), len(Dataset)))
    min = np.min(Dataset)
    max = np.max(Dataset)
    for i in range(0, len(Dataset)):
        for j in range(0, len(Dataset)):
            LinearTrans[i][j] = 0 + ((Dataset[i][j] - min) / (max - min)) * 255
    return LinearTrans


def histEqualization(filename):  # Function for Histogram Equalization
    file = open(filename, "r")
    data = file.readlines()

    dataset = np.zeros(shape=(len(data), len(data)))
    for i in range(len(data)):
        data[i] = re.sub("[\"\n]", '', data[i])
        a = data[i].split(",")
        for j in range(len(data)):
            dataset[i][j] = int(np.log(10 + float(a[j])) * 10000)

    dataset = np.flip(dataset, axis=0)
    Array = np.reshape(dataset, 250000)

    hist = [0 for i in range(0, int(np.max(Array)) + 1)]
    transform = [0 for i in range(0, int(np.max(Array)) + 1)]

    for i in Array:
        hist[int(i)] += 1
        transform[int(i)] += 1

    for i in range(len(transform)):
        transform[i] = transform[i] + transform[i - 1]

    for i in range(len(transform)):
        transform[i] = int((transform[i] / len(Array)) * 255)

    for i in range(len(Array)):
        Array[i] = transform[int(Array[i])]

    resultDataset = np.reshape(Array, (500, 500))

    return resultDataset


# # (a) Calculate the max value, the min value, the mean value and the variance value of this 2D data set.

file = open("i170b2h0_t0.txt", "r")
data = file.readlines()

Matrix = np.zeros(shape=(len(data), len(data)))
Matrix1 = np.zeros(shape=(len(data), len(data)))

for i in range(len(data)):
    data[i] = re.sub("[\"\n]", '', data[i])
    a = data[i].split(",")
    for j in range(len(data)):
        Matrix1[i][j] = int(np.log(1 + float(a[j])) * 10000)
        Matrix[i][j] = float(a[j])

Matrix = np.flip(Matrix, axis=0)
Matrix1 = np.flip(Matrix1, axis=0)


Array1D = np.reshape(Matrix, 250000)

print("Max Value of 2D Dataset", np.max(Matrix))
print("Min Value of 2D Dataset", np.min(Matrix))

Mean = np.sum(Array1D) / len(Array1D)
print("Mean: ", Mean)

var = 0
for i in Array1D:
    var += (i - Mean) ** 2

Variance = var / len(Array1D)
print("Variance: ", Variance)

# (b) Draw a profile line through the line with the maximum value of this 2D dataset; you will need
# coordinate axes to read off values

x, y = (np.where(Matrix == (np.max(Matrix))))
ax1, fig1 = plt.subplots()
fig1.plot(Matrix[int(x)])
fig1.set_title("Profile Line through the line with the maximum value of this 2D dataset")
ax1.savefig('b.Profile_Line.png')

# (c) Display a histogram of this 2D data set (instead of bars you may use a line graph to link occurrences
# along the x axis)

Matrix1 = linearTransformation(Matrix1)

Array = np.reshape(Matrix1, 250000)
hist = [0 for i in range(int(np.min(Array)), int(np.max(Array)) + 2)]
for i in Array:
    hist[int(i)] += 1
ax2, fig2 = plt.subplots()
fig2.plot(hist)
plt.ylabel("Occurrence")
plt.xlabel("Data Values")
fig2.set_title("Histogram of the 2D Dataset(non linearly transformed with log)")
ax2.savefig("c.Histogram(lineGraph).png")

hist1 = [0 for i in range(int(np.min(Array1D)), int(np.max(Array1D)) + 2)]
for i in Array1D:
    hist1[int(i)] += 1
ax9, fig9 = plt.subplots()
fig9.plot(hist1)
plt.ylabel("Occurrence")
plt.xlabel("Data Values")
fig9.set_title("Histogram of 2D DataSet with raw data")
ax9.savefig("c.Histogram(Raw Data).png")

# (d) Rescale values to range between 0 and 255 using your own transformation and display on your screen.

# Matrix1 is transformed already at line 98

ax3, fig3 = plt.subplots()
fig3.imshow(Matrix1, cmap="gray")
fig3.set_title("Rescale between 0 and 255 using log transformation")
ax3.savefig("d.TransformedImage.png")
# #plt.show()

# (e) Carry out a Histogram Equalization on each of the four bands and display on your screen

result1 = histEqualization("i170b1h0_t0.txt")
ax4, fig4 = plt.subplots()
fig4.imshow(result1, cmap="gray")
fig4.set_title("Band b1")
ax4.savefig("e1.HistogramEqualization B1.png")

result2 = histEqualization("i170b2h0_t0.txt")
ax5, fig5 = plt.subplots()
fig5.imshow(result2, cmap="gray")
fig5.set_title("Band b2")
ax5.savefig("e2.HistogramEqualization B2.png")

result3 = histEqualization("i170b3h0_t0.txt")
ax6, fig6 = plt.subplots()
fig6.imshow(result3, cmap="gray")
fig6.set_title("Band b3")
ax6.savefig("e3.HistogramEqualization B3.png")

result4 = histEqualization("i170b4h0_t0.txt")
ax7, fig7 = plt.subplots()
fig7.imshow(result4, cmap="gray")
fig7.set_title("Band b4")
ax7.savefig("e4.HistogramEqualization B4.png")

# (f) Combine the histo-equalized data set to an RGB-image

RGBImage = np.zeros((len(result4), len(result4), 3), "uint8")
RGBImage[..., 0] = result4
RGBImage[..., 1] = result3
RGBImage[..., 2] = result1

ax8, fig8 = plt.subplots()
fig8.imshow(RGBImage)
fig8.set_title("RGB Image")
ax8.savefig("f.RGBImage.png")
plt.show()
