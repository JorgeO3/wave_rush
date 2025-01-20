/// Macro para crear vectores con alineación de memoria personalizada.
///
/// Esta macro proporciona diferentes formas de crear vectores alineados en memoria,
/// permitiendo especificar el tamaño, alineación, tipo, valores por defecto y capacidad.
///
/// ### Casos de Uso
///
/// ```rust
/// let vec = aligned_vec!(10, 16);  // Vector de 10 elementos alineado a 16 bytes
/// ```
///
/// ```rust
/// let vec: Vec<i32> = aligned_vec!(i32, 5, 16);  // Vector de 5 i32 alineado a 16 bytes
/// ```
///
/// ```rust
/// let vec = aligned_vec!(i32, 3, 16, 42);  // Vector de 3 i32 inicializados a 42 alineado a 16 bytes
/// ```
///
/// ```rust
/// let vec = aligned_vec!(10, 16, capacity = 20);  // Vector len 10, capacidad 20
/// ```
///
/// ### Argumentos
///
/// Dependiendo del caso de uso, la macro acepta diferentes combinaciones de argumentos:
///
/// * `$size:expr`: Número de elementos en el vector
/// * `$align:expr`: Alineación deseada en bytes (debe ser potencia de 2)
/// * `$type:ty`: Tipo de datos para el vector
/// * `$default:expr`: Valor por defecto para inicializar elementos
/// * `capacity = $cap:expr`: Capacidad inicial del vector
///
/// ### Seguridad
///
/// Esta macro usa código unsafe internamente para manejar la memoria alineada.
/// Es responsabilidad del usuario asegurar que:
/// * La alineación sea una potencia de 2
/// * El tamaño total no exceda isize::MAX bytes
/// * La memoria sea liberada correctamente
///
/// ### Ejemplos
///
/// ```rust
/// use std::alloc::{alloc, Layout};
///
/// // Ejemplo 1: Vector básico alineado
/// let mut vec1 = aligned_vec!(10, 16);
/// assert_eq!(vec1.len(), 10);
///
/// // Ejemplo 2: Vector de enteros alineado
/// let mut vec2: Vec<i32> = aligned_vec!(i32, 5, 16);
/// vec2[0] = 42;
/// assert_eq!(vec2.len(), 5);
///
/// // Ejemplo 3: Vector inicializado
/// let vec3 = aligned_vec!(i32, 3, 16, 42);
/// assert_eq!(vec3[0], 42);
/// assert_eq!(vec3[1], 42);
/// assert_eq!(vec3[2], 42);
///
/// // Ejemplo 4: Vector con capacidad extra
/// let mut vec4 = aligned_vec!(10, 16, capacity = 20);
/// assert_eq!(vec4.len(), 10);
/// assert_eq!(vec4.capacity(), 20);
/// ```
#[macro_export]
macro_rules! aligned_vec {
    // Caso 1: Vector básico (tamaño y alineación)
    ($size:expr, $align:expr) => {{
        let size = $size;
        let layout = Layout::from_size_align(size, $align).unwrap();
        let aligned_ptr = unsafe { alloc(layout) };
        if aligned_ptr.is_null() {
            std::alloc::handle_alloc_error(layout)
        }
        unsafe { Vec::from_raw_parts(aligned_ptr as *mut _, size, size) }
    }};

    // Caso 2: Vector con tipo específico
    ($type:ty, $size:expr, $align:expr) => {{
        let size = $size * std::mem::size_of::<$type>();
        let layout = Layout::from_size_align(size, $align).unwrap();
        let aligned_ptr = unsafe { alloc(layout) };
        if aligned_ptr.is_null() {
            std::alloc::handle_alloc_error(layout)
        }
        unsafe { Vec::from_raw_parts(aligned_ptr as *mut $type, $size, $size) }
    }};

    // Caso 3: Vector con valor por defecto
    ($type:ty, $size:expr, $align:expr, $default:expr) => {{
        let mut vec = aligned_vec!($type, $size, $align);
        for i in 0..$size {
            vec[i] = $default;
        }
        vec
    }};

    // Caso 4: Vector con capacidad específica
    ($size:expr, $align:expr, capacity = $cap:expr) => {{
        let size = $size;
        let capacity = $cap;
        assert!(
            capacity >= size,
            "La capacidad debe ser mayor o igual al tamaño"
        );
        let layout = Layout::from_size_align(capacity, $align).unwrap();
        let aligned_ptr = unsafe { alloc(layout) };
        if aligned_ptr.is_null() {
            std::alloc::handle_alloc_error(layout)
        }
        unsafe { Vec::from_raw_parts(aligned_ptr as *mut _, size, capacity) }
    }};
}

// Tests
#[cfg(test)]
mod tests {
    use super::*;
    use std::alloc::{alloc, Layout};

    #[test]
    fn test_aligned_vec_basic() {
        let vec = aligned_vec!(10, 16);
        assert_eq!(vec.len(), 10);
    }

    #[test]
    fn test_aligned_vec_typed() {
        let vec: Vec<i32> = aligned_vec!(i32, 5, 16);
        assert_eq!(vec.len(), 5);
    }

    #[test]
    fn test_aligned_vec_default() {
        let vec: Vec<i32> = aligned_vec!(i32, 3, 16, 42);
        assert!(vec.iter().all(|&x| x == 42));
    }

    #[test]
    fn test_aligned_vec_capacity() {
        let vec: Vec<u8> = aligned_vec!(10, 16, capacity = 20);
        assert_eq!(vec.len(), 10);
        assert_eq!(vec.capacity(), 20);
    }

    #[test]
    #[should_panic(expected = "La capacidad debe ser mayor o igual al tamaño")]
    fn test_invalid_capacity() {
        let _vec: Vec<u8> = aligned_vec!(10, 16, capacity = 5);
    }
}
