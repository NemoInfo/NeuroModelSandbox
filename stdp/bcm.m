x1 = [20; 0];
x2 = [0; 20];
p  = 0.5;

dt  = 1;      % ms
T   = 4e4/dt; % ms
eta = 1e-6;   % ms^-1
y0  = 10;
tau = 50;     % ms

tht  = zeros(T, 1);
y    = zeros(T, 1);
w    = zeros(T, 2);
w(1,:) = [0.5 0.5];

for t = 2:T
  x = x1;
  if rand() > p, x = x2; end
  
  y(t)    = w(t-1,:) * x;
  tht(t)  = tht(t-1)  + dt / tau * (-tht(t-1) + y(t)^2 / y0);
  w(t, :) = w(t-1, :) + eta * x' * (y(t) - tht(t));
  w(t, :) = max(w(t, :), 0);
end

figure;
subplot 311; hold on;
plot(w(:,1));
plot(w(:,2));
legend(["w1" "w2"])

subplot 312;
plot(y);
legend("y");

subplot 313;
plot(tht);
legend("theta");