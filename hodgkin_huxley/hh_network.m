function hh_network(options)
    arguments 
        options.A double = [0 0.2 -0.2; -0.2 0 0.3; -0.2 0.1 0] % Connection matrix
        options.I_ext function_handle = @(t, i) ... ; % External current
            (i == 3) * (10 * (t > 5 & t < 60)) + ... 
            (i == 1) * (10 * (t > 40 & t < 60));
    end
    A = options.A;
    N = size(A, 1);    % Number of neurons
    
    % Initial conditions
    V0 = -65 * ones(N, 1); % Resting membrane potential (mV)
    m0 = alpha_m(V0) ./ (alpha_m(V0) + beta_m(V0));
    n0 = alpha_n(V0) ./ (alpha_n(V0) + beta_n(V0));
    h0 = alpha_h(V0) ./ (alpha_h(V0) + beta_h(V0));
    y0 = [V0; m0; n0; h0]; % Initial state vector as a single column
    I_ext = options.I_ext;

    % Time span
    tspan = [0 100]; % Time in ms

    [t, y] = ode45(@(t, y) hh_network_odes(t, y, N, A, I_ext), tspan, y0);

    for i = 1:N
        subplot(N * 2, 1, i * 2 - 1);
        plot(t, y(:, i), 'LineWidth', 1.5);
        ylabel(['$u$ (mV)'], 'Interpreter','latex', FontSize=14);
        title(['$\textbf{U}_{', num2str(i), '}$'], 'Interpreter','latex', FontSize=20);
        grid on;
        subplot(N * 2, 1, i * 2);
        plot(t, arrayfun(@(tt) I_ext(tt, i), t), 'r', 'LineWidth',1.5);
        ylabel('I ($nA$)', 'Interpreter','latex', FontSize=14);
        title(['$I_{ext}^{', num2str(i),'}$'], 'Interpreter','latex', FontSize=20);
    end
    xlabel('t (ms)', FontSize=14);
end

function dydt = hh_network_odes(t, y, N, A, I_ext_func)
    % Hodgkin-Huxley parameters
    C_m = 1;        % Membrane capacitance (uF/cm^2)
    g_Na = 120;     % Sodium conductance (mS/cm^2)
    g_K = 36;       % Potassium conductance (mS/cm^2)
    g_L = 0.3;      % Leak conductance (mS/cm^2)
    E_Na = 50;      % Sodium reversal potential (mV)
    E_K = -77;      % Potassium reversal potential (mV)
    E_L = -54.4;    % Leak reversal potential (mV)
    E_syn = 0; % Synaptic reversal potential (mV)

    % Split state vector
    V = y(1:N); % Membrane potentials
    m = y(N + (1:N)); % Sodium activation
    h = y(2 * N + (1:N)); % Sodium inactivation
    n = y(3 * N + (1:N)); % Potassium activation

    % Compute external current
    I_ext = arrayfun(@(neuron) I_ext_func(t, neuron), 1:N)';

    % Synaptic current
    I_syn = A * (V - E_syn);

    % Hodgkin-Huxley equations for each neuron
    dm = alpha_m(V) .* (1 - m) - beta_m(V) .* m;
    dh = alpha_h(V) .* (1 - h) - beta_h(V) .* h;
    dn = alpha_n(V) .* (1 - n) - beta_n(V) .* n;

    % Ionic currents
    I_Na = g_Na .* (m .^ 3) .* h .* (V - E_Na);
    I_K = g_K .* (n .^ 4) .* (V - E_K);
    I_L = g_L .* (V - E_L);

    % Membrane potential dynamics
    dV = (I_ext - I_Na - I_K - I_L - I_syn) / C_m;

    % Combine into a single column vector
    dydt = [dV; dm; dh; dn];
end

% Vectorized rate functions
function val = alpha_m(V), val = (2.5 - 0.1 * (V + 65)) ./ (exp(2.5 - 0.1 * (V + 65)) - 1); end
function val = beta_m(V), val = 4 * exp(-(V + 65) / 18); end
function val = alpha_n(V), val = (0.1 - 0.01 * (V + 65)) ./ (exp(1 - 0.1 * (V + 65)) - 1); end
function val = beta_n(V), val = 0.125 * exp(-(V + 65) / 80); end
function val = alpha_h(V), val = 0.07 * exp(-(V + 65) / 20); end
function val = beta_h(V), val = 1 ./ (exp(3 - 0.1 * (V + 65)) + 1); end
