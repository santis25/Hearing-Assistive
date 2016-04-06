%adding noise to sound wave files


for n = 1:1:100
    %load file
    [y,Fs] = audioread('School_Fire_Alarm.wav');
    %set STD
    STD = n /1000;
    
    z = y + STD*randn(size(y));
    filename = sprintf('School_Fire_Vars/School_Wav_Noise_%d.wav',n);
    audiowrite(filename, z, Fs)

    
end
%Other methods of adding noise
z = imnoise( y, 'salt & pepper', 0.05);
z = imnoise( y, 'poisson');
z = imnoise( y, 'gaussian', Mean, Var);