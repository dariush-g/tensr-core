use tensr_core::tensor::Tensor;

#[test]
fn test_add() -> Result<(), Box<dyn std::error::Error>> {
    let a = Tensor::from_data(vec![2, 2], vec![1, 2, 3, 4])?;
    let b = Tensor::from_data(vec![2, 2], vec![5, 6, 7, 8])?;

    let result = a + b;

    assert_eq!(result.shape, vec![2, 2]);
    assert_eq!(result.data, vec![6, 8, 10, 12]);

    Ok(())
}

#[test]
fn test_mul() -> Result<(), Box<dyn std::error::Error>> {
    let a = Tensor::from_data(vec![2, 2], vec![1, 2, 3, 4])?;
    let b = Tensor::from_data(vec![2, 2], vec![5, 6, 7, 8])?;

    let result = a.matmul(&b);

    assert_eq!(result.unwrap().data, vec![19, 22, 43, 50]);

    Ok(())
}

#[test]
fn test_sub() -> Result<(), Box<dyn std::error::Error>> {
    let a = Tensor::from_data(vec![2, 2], vec![1, 2, 3, 4])?;
    let b = Tensor::from_data(vec![2, 2], vec![5, 6, 7, 8])?;

    let result = a - b;

    assert_eq!(result.shape, vec![2, 2]);
    assert_eq!(result.data, vec![-4, -4, -4, -4]);

    Ok(())
}
