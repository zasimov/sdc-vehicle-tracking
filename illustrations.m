clear all;

load metrics.mat;

p = figure();
hold on;

plot(carn)
title('The Number of Detected Vehicles')
xlabel("frame #")
ylabel("number of vehicles")

print(p, "output_images/metrics/carn.png", '-dpng');




p = figure();
hold on;

plot(windowsn)
title('The Number of Hot Windows')
xlabel("frame #")
ylabel("number of hot windows")

print(p, "output_images/metrics/windowsn.png", '-dpng');


p = figure();
hold on;

plot(heatmax)
title('The Max value of Heat Map')
xlabel("frame #")
ylabel("heat max")

print(p, "output_images/metrics/heatmax.png", '-dpng');