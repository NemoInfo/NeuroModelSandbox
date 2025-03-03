rng(69)
% Defining network model parameters
vt = 1;                     % Spiking threshold
tau_m = 10;                 % Membrane time constant [ms]
g_m = 1;                    % Neuron conductance
Nsig = 0.25;                 % Variance amplitude of current
Nmean = 0.75;                % Mean current to neurons
tau_I = 10;                 % Time constant to filter the synaptic inputs
N = 5000;                   % Number of neurons in total
NE = 0.5*N;                 % Number of excitatory neurons
NI = 0.5*N;                 % Number of inhibitory neurons
dt = 1;                     % Simulation time bin [ms]
T = 300/dt;                 % Simulation length 
W = 100/N;                  % Connectivity strength

% Initialization

v = rand(N,1)*vt;           % membrane potential
vv = zeros(N,1);            % variable that notes if v crosses the threshold
Iback = zeros(N,1);         % building up the external current
SP = 0;                     % recording spike times
Ichem = zeros(N,1);         % current coming from synaptic inputs
Iext = zeros(N,1);     % external current
raster = [];                % save spike times for plotting

I_chem_over_time = zeros(T, 1); 
% loop over the time
for t = 1:T
    Iback = Iback + dt/tau_I*(-Iback +randn(N,1));          % generate a colored noise for the current
    Iext = Iback/sqrt(1/(2*(tau_I/dt)))*Nsig+Nmean;         % rescaling the noise current to have the correct mean and variance

    Ichem(1:NE) = Ichem(1:NE) + dt/tau_I*(-Ichem(1:NE) + W*(sum(vv(1:NE))-vv(1:NE))-W*(sum(vv(NE+1:end)))); % current to excitatory neurons coming from the synaptic inputs
    Ichem(NE+1:end) = Ichem(NE+1:end) + dt/tau_I*(-Ichem(NE+1:end) -W*(sum(vv(NE+1:end))-vv(NE+1:end))+W*(sum(vv(1:NE)))); % current to inhibitory neurons coming from the synaptic inputs
    I_chem_over_time(t) = Ichem(69);
    Itot = Iext+Ichem;
    %%%%%%%%%%% To insert integrate-and-fire model here  %%%%%%%%%%%%%
    v = v + dt / tau_m * (-v + Itot / g_m);
    if t == 200
      v(end-300:end) = vt;
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    vv =(v>=vt);                                        % spike if voltage crosses the threshold    
    
    v = (1-vv).*v;                                      % reset after spike
    SP = find(vv);                                      % find the spike times
    raster=[raster;t*ones(length(SP),1),SP];            % save spike times for plotting
end

% Plot the raster output
h = figure; hold on;
plot(raster(:,1)*dt, raster(:,2),'.b')
xlim([100 300])
xlabel('time [ms]','fontsize',20)
ylabel('neuron index','fontsize',20)
set(gca,'fontsize',20);
set(gca,'YDir','normal')
figure;
plot(I_chem_over_time);