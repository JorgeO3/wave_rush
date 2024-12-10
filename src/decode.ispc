export void decode_int16_avx2(const uniform int16 buffer_i16[], 
                               const uniform int total_samples, 
                               uniform int packets[]) {
    // Procesamiento vectorial nativo de ISPC
    foreach (i = 0 ... total_samples) {
        // Conversión de int16 a int usando casting explícito
        packets[i] = (int)buffer_i16[i];
    }
}