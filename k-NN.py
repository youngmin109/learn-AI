import matplotlib.pyplot as plt

weather_Frogsound1 = [70, 60, 90, 80, 50, 80, 90, 70, 80, 100,
                     40, 30, 60, 20, 50, 0, 0, 0, 0, 0]
weather_KneelPain1 = [44, 39, 49, 36, 48, 29, 39, 43, 41, 44,
                    38, 44, 49, 39, 48, 42, 47, 42, 38, 36]

plt.scatter(weather_Frogsound1, weather_KneelPain1)
plt.xlabel('FrogSound')
plt.ylabel('KneePain')
plt.show