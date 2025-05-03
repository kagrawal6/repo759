

% Read data, skip header
data = dlmread('timing1.txt', ' ', 1, 0);

% Extract columns
n  = data(:,1);
t1 = data(:,2);   % matmul_1 (int)
t2 = data(:,3);   % matmul_2 (float)
t3 = data(:,4);   % matmul_3 (double)

% Create figure
figure;
plot(n, t1, '-o', 'LineWidth', 1.5); hold on;
plot(n, t2, '-s', 'LineWidth', 1.5);
plot(n, t3, '-^', 'LineWidth', 1.5);
hold off;

% Labels and legend
xlabel('n (matrix dimension)');
ylabel('Time (ms)');
legend('matmul 1 (int)', 'matmul 2 (float)', 'matmul 3 (double)', ...
       'Location', 'northwest');
grid on;
title('Task 1: Matrix Multiplication Timings');

% Save as PDF
set(gcf, 'PaperPositionMode', 'auto');
print(gcf, 'task1', '-dpdf');
