# Contributing to Pylon

This document provides guidelines for contributing to the Pylon coordination infrastructure project.

## Table of Contents

- [Development Environment](#development-environment)
- [Code Organization](#code-organization)
- [Contribution Workflow](#contribution-workflow)
- [Code Standards](#code-standards)
- [Testing Requirements](#testing-requirements)
- [Documentation Standards](#documentation-standards)
- [Security Considerations](#security-considerations)

## Development Environment

### Prerequisites

- Rust 1.75.0 or higher
- Python 3.11+ (for analysis components)
- Node.js 18+ (for web interface components)
- Git with LFS support
- Docker (for containerized development)

### Setup

```bash
# Clone the repository
git clone https://github.com/organization/pylon.git
cd pylon

# Setup development environment
make setup

# Verify installation
make check
```

### Development Dependencies

Install required development tools:

```bash
# Install cargo tools
cargo install cargo-watch cargo-tarpaulin cargo-audit
cargo install cargo-deny cargo-machete cargo-outdated

# Install Rust components
rustup component add clippy rustfmt
rustup target add wasm32-unknown-unknown
```

## Code Organization

### Crate Structure

```
crates/
├── pylon-core/              # Core coordination engine
├── pylon-coordinator/       # Main coordinator service
├── cable-network/           # Network temporal coordination
├── cable-spatial/           # Autonomous spatial navigation
├── cable-individual/        # Individual experience optimization
├── temporal-economic/       # Economic convergence engine
├── precision-by-difference/ # Core mathematics library
├── pylon-sdk/               # Client SDKs
├── pylon-cli/               # Command-line interface
├── pylon-web/               # Web interface backend
├── pylon-metrics/           # Metrics and monitoring
├── pylon-config/            # Configuration management
└── pylon-test-utils/        # Testing utilities
```

### Module Guidelines

- Each crate should have a clear, single responsibility
- Public APIs should be well-documented with examples
- Internal modules should be organized logically
- Dependencies between crates should be minimal and well-justified

## Contribution Workflow

### 1. Issue Creation

Before implementing changes:

- Check existing issues to avoid duplication
- Create an issue describing the problem or enhancement
- Include relevant context and motivation
- Tag appropriately (bug, enhancement, documentation, etc.)

### 2. Branch Strategy

```bash
# Create feature branch from main
git checkout main
git pull origin main
git checkout -b feature/description

# For bug fixes
git checkout -b fix/issue-number

# For documentation
git checkout -b docs/description
```

### 3. Development Process

```bash
# Make changes following code standards
# Write tests for new functionality
# Update documentation as needed

# Run quick development cycle
make quick

# Run full validation before submission
make full
```

### 4. Pull Request Process

1. Push branch to origin
2. Create pull request with descriptive title and body
3. Link related issues
4. Request review from maintainers
5. Address feedback and update as needed
6. Maintain clean commit history

### Pull Request Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes (or properly documented)
```

## Code Standards

### Rust Code Style

Follow standard Rust conventions with project-specific additions:

```toml
# clippy.toml configuration is enforced
cognitive-complexity-threshold = 25
missing-docs-in-public-items = true
too-many-arguments-threshold = 8
```

### Formatting

```bash
# Format code before committing
cargo fmt --all

# Check formatting
cargo fmt --all -- --check
```

### Linting

```bash
# Run clippy with project configuration
cargo clippy --workspace --all-features -- -D warnings
```

### Naming Conventions

- **Crates**: `kebab-case` (e.g., `pylon-core`)
- **Modules**: `snake_case` (e.g., `temporal_coordination`)
- **Functions**: `snake_case` (e.g., `calculate_precision`)
- **Types**: `PascalCase` (e.g., `CoordinationFragment`)
- **Constants**: `SCREAMING_SNAKE_CASE` (e.g., `DEFAULT_PRECISION`)

### Error Handling

- Use `Result<T, E>` for fallible operations
- Create specific error types using `thiserror`
- Provide meaningful error messages
- Document error conditions in function documentation

```rust
use thiserror::Error;

#[derive(Error, Debug)]
pub enum CoordinationError {
    #[error("Precision calculation failed: {reason}")]
    PrecisionCalculation { reason: String },
    
    #[error("Invalid configuration: {0}")]
    InvalidConfiguration(String),
}
```

### Documentation

All public APIs must include:

- Purpose and behavior description
- Parameter documentation
- Return value description
- Example usage
- Error conditions

```rust
/// Calculates precision-by-difference for temporal coordination.
///
/// This function computes the difference between a reference temporal
/// coordinate and local measurement, enabling coordination enhancement.
///
/// # Arguments
///
/// * `reference` - Absolute temporal reference coordinate
/// * `local` - Local temporal measurement
///
/// # Returns
///
/// Returns the precision enhancement vector or an error if calculation fails.
///
/// # Examples
///
/// ```rust
/// use pylon_core::calculate_temporal_precision;
/// 
/// let precision = calculate_temporal_precision(ref_time, local_time)?;
/// ```
///
/// # Errors
///
/// Returns [`CoordinationError::PrecisionCalculation`] if input values are invalid.
pub fn calculate_temporal_precision(
    reference: TemporalCoordinate,
    local: TemporalCoordinate,
) -> Result<PrecisionVector, CoordinationError> {
    // Implementation
}
```

## Testing Requirements

### Test Categories

1. **Unit Tests**: Test individual functions and modules
2. **Integration Tests**: Test component interactions
3. **Property Tests**: Test invariants using `proptest`
4. **Benchmark Tests**: Performance validation using `criterion`

### Test Organization

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use pylon_test_utils::*;

    #[test]
    fn test_precision_calculation() {
        // Arrange
        let reference = create_test_temporal_coordinate();
        let local = create_test_local_measurement();

        // Act
        let result = calculate_temporal_precision(reference, local);

        // Assert
        assert!(result.is_ok());
        let precision = result.unwrap();
        assert!(precision.magnitude > 0.0);
    }

    #[tokio::test]
    async fn test_async_coordination() {
        // Test async functionality
    }
}
```

### Property Testing

Use `proptest` for testing invariants:

```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn precision_calculation_invariants(
        ref_val in 0.0f64..1000.0,
        local_val in 0.0f64..1000.0
    ) {
        let reference = TemporalCoordinate::new(ref_val);
        let local = TemporalCoordinate::new(local_val);
        
        let precision = calculate_temporal_precision(reference, local)?;
        
        // Invariant: precision magnitude should be finite
        prop_assert!(precision.magnitude.is_finite());
    }
}
```

### Test Utilities

Use shared test utilities from `pylon-test-utils`:

```rust
use pylon_test_utils::{
    create_test_coordinator,
    create_test_temporal_session,
    assert_precision_within_bounds,
};

#[tokio::test]
async fn test_coordinator_functionality() {
    let coordinator = create_test_coordinator().await;
    let session = create_test_temporal_session();
    
    let result = coordinator.coordinate(&session).await?;
    
    assert_precision_within_bounds(&result, 1e-9);
}
```

## Documentation Standards

### Code Documentation

- Document all public APIs
- Include examples for complex functionality
- Explain non-obvious implementation decisions
- Reference relevant academic papers or specifications

### User Documentation

- README files for each major component
- API documentation generated with `cargo doc`
- Integration guides and examples
- Configuration reference documentation

### Academic Documentation

Given the theoretical foundation of Pylon, maintain academic rigor in documentation:

- Reference theoretical frameworks
- Include mathematical derivations where relevant
- Cite academic sources appropriately
- Maintain consistency with published papers

## Security Considerations

### Security Guidelines

- Never commit secrets or credentials
- Use secure defaults in configuration
- Validate all inputs at system boundaries
- Follow principle of least privilege
- Implement proper error handling without information leakage

### Security Review Process

All security-relevant changes require:

1. Security-focused code review
2. Threat modeling consideration
3. Security testing validation
4. Documentation of security implications

### Reporting Security Issues

Report security vulnerabilities privately to: security@pylon-coordination.org

Do not open public issues for security vulnerabilities.

## Release Process

### Version Management

Pylon uses semantic versioning (SemVer):

- **Patch** (0.0.x): Bug fixes and minor improvements
- **Minor** (0.x.0): New features, backward compatible
- **Major** (x.0.0): Breaking changes

### Release Checklist

Before releasing:

- [ ] All tests pass
- [ ] Documentation updated
- [ ] Security audit completed
- [ ] Performance benchmarks run
- [ ] Breaking changes documented
- [ ] Migration guide provided (if needed)

## Communication

### Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and discussions
- **Pull Requests**: Code review and technical discussion

### Code Review Guidelines

#### For Authors

- Keep changes focused and atomic
- Write clear commit messages
- Include tests for new functionality
- Update documentation as needed
- Respond promptly to review feedback

#### For Reviewers

- Focus on correctness, clarity, and maintainability
- Check for proper error handling
- Verify test coverage
- Ensure documentation is accurate
- Be constructive and respectful in feedback

### Commit Message Format

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Formatting changes
- `refactor`: Code restructuring
- `test`: Adding tests
- `chore`: Maintenance tasks

Examples:
```
feat(cable-network): implement temporal fragmentation protocol

Add support for distributing network messages across temporal 
coordinates for enhanced coordination precision.

Closes #123
```

## Performance Considerations

### Benchmarking

- Include benchmarks for performance-critical code
- Use `criterion` for statistical benchmarking
- Document performance characteristics
- Monitor for regressions in CI

### Optimization Guidelines

- Measure before optimizing
- Focus on algorithmic improvements first
- Consider memory allocation patterns
- Profile production workloads when possible

## Legal Considerations

### Licensing

All contributions are subject to the MIT license. By contributing, you agree that your contributions will be licensed under the same terms.

### Intellectual Property

Ensure that contributions:

- Are your original work or properly attributed
- Do not violate any third-party copyrights
- Include appropriate license headers where required

---

## Getting Help

If you need assistance:

1. Check existing documentation and issues
2. Ask questions in GitHub Discussions
3. Join community discussions
4. Contact maintainers directly for complex issues

Thank you for contributing to Pylon!
