# 🛠️ Haive Development Utilities

**A unified, powerful, and easy-to-use debugging, logging, tracing, and profiling system for Python development.**

## 🚀 Quick Start

```python
# One import gets you everything
from haive.core.utils.dev import debug, log, trace, profile, benchmark

# Enhanced debugging (icecream replacement)
debug.ice("Hello", variable=42)  # 🍦 Beautiful variable inspection

# Rich logging with context
with log.context("my_operation"):
    log.info("Starting process...")
    log.success("Complete!")  # ✅

# Performance profiling
@profile.time
def my_function():
    return "profiled!"

# Function tracing
@trace.calls
def traced_function():
    return "traced!"
```

## 🎯 What You Get

### 🐛 **Enhanced Debugging**

- **icecream replacement** with rich formatting
- **Web-based debugging** in your browser
- **Visual debugging** with pudb integration
- **Auto-breakpoints** on exceptions
- **Call tracing** with birdseye integration

### 📋 **Rich Logging**

- **Beautiful terminal output** with Rich integration
- **Structured logging** with context managers
- **Progress indicators** and status messages
- **Data visualization** (tables, JSON, metrics)
- **Timing utilities** and performance logging

### 🔍 **Code Tracing**

- **Function call tracking** with detailed statistics
- **Variable change monitoring** with history
- **Advanced tracing** with pysnooper and hunter
- **Call stack analysis** and debugging
- **Trace reports** with markdown export

### ⚡ **Performance Profiling**

- **Function timing** with statistical analysis
- **Memory profiling** (line-by-line and total usage)
- **CPU profiling** with pyinstrument and scalene
- **Comprehensive profiling** (all metrics at once)
- **Performance comparison** between functions

### 🧪 **Benchmarking & Stress Testing**

- **Precise timing** with statistical analysis
- **Load testing** with concurrent users
- **Spike testing** for performance limits
- **Async benchmarking** for async functions
- **Historical tracking** and visualization

## 📖 Usage Examples

### Simple Debugging

```python
def calculate_tax(income, rate):
    debug.ice(income, rate)  # See inputs

    tax = income * rate
    debug.ice(tax)  # See result

    return tax
```

### Rich Logging Workflow

```python
def process_orders(orders):
    log.info("Processing orders", count=len(orders))

    with log.context("validation"):
        log.progress("Validating orders...")
        # validation code
        log.success("Orders validated!")

    with log.context("processing"):
        for i, order in enumerate(orders):
            if i % 10 == 0:
                log.progress(f"Processed {i}/{len(orders)}")
        log.success("All orders processed!")

    log.metrics({
        "total_orders": len(orders),
        "processing_time": "2.5s",
        "success_rate": "98.5%"
    })
```

### Performance Analysis

```python
@profile.comprehensive  # Times, memory, CPU profiling
@trace.calls           # Track all function calls
def data_processor(dataset):
    with log.context("data_processing"):
        debug.ice("Processing dataset", size=len(dataset))

        # Your processing code here
        result = heavy_computation(dataset)

        log.success("Processing complete!")
        return result

# Benchmark the function
benchmark.time_it(data_processor, test_data, iterations=10)
```

### Unified Development Workflow

```python
@debug.breakpoint_on_exception  # Auto-debug on errors
@trace.snoop                   # Detailed execution tracing
@profile.time                  # Performance timing
def complex_algorithm(data):
    with log.context("algorithm_execution"):
        log.info("Algorithm started", data_size=len(data))

        # Track variables as they change
        trace.vars(input_size=len(data))

        # Debug key steps
        debug.ice("Starting phase 1")
        phase1_result = phase_one(data)
        trace.vars(phase1_complete=True)

        debug.ice("Starting phase 2")
        phase2_result = phase_two(phase1_result)
        trace.vars(phase2_complete=True)

        log.success("Algorithm complete!")
        return phase2_result

# Get comprehensive statistics
trace.stats()    # Call statistics
profile.stats()  # Performance statistics
```

## 🎮 Usage Modes

### 🏃‍♂️ **Quick & Dirty Mode**

```python
debug.ice(my_variable)           # Just see the value
log.info("Something happened")   # Quick logging
```

### 🔧 **Development Mode**

```python
with log.context("debug_session"):
    debug.ice(data)
    trace.vars(x=x, y=y, state="processing")
    log.progress("Debugging complex issue...")

    with profile.profile_context("performance_check"):
        result = expensive_operation()
```

### 🚀 **Production Monitoring**

```python
with log.context("production_task") as ctx:
    log.info("Task started", user_id=user.id)

    with log.timer("database_operation"):
        data = fetch_from_database()

    with profile.profile_context("computation"):
        result = process_data(data)

    log.success("Task completed",
                duration=ctx.duration,
                result_count=len(result))
```

## 🛡️ Smart Fallbacks

**Works everywhere, even without optional dependencies:**

```python
# Without icecream -> falls back to rich print -> falls back to basic print
debug.ice("Still works!")

# Without rich -> falls back to standard logging
log.info("Beautiful logging!")

# Without profiling tools -> falls back to basic timing
profile.time(my_function)

# Without tracing tools -> falls back to simple call tracking
trace.calls(my_function)
```

## 🔧 Advanced Features

### Web-Based Debugging

```python
debug.web(port=5556)  # Debug in browser at localhost:5556
```

### Visual Debugging

```python
debug.visual()  # Full-screen debugger interface
```

### Advanced Tracing

```python
# Hunter-based advanced tracing
trace.hunt("call", action="return", module="mymodule")

# Generate detailed reports
report = trace.report("debug_session.md")
```

### Comprehensive Benchmarking

```python
# Add multiple benchmarks
benchmark.add_benchmark("fast_algo", fast_function, test_data)
benchmark.add_benchmark("slow_algo", slow_function, test_data)

# Run complete benchmark suite
results = benchmark.run_suite()

# Compare with historical data
comparison = benchmark.compare_with_history("fast_algo")

# Plot results (requires matplotlib)
benchmark.plot_results("performance_trends")
```

### Load Testing

```python
# Stress test your functions
results = benchmark.stress_tester.load_test(
    my_api_function,
    concurrent_users=50,
    duration_seconds=60,
    ramp_up_seconds=10
)

# Spike testing
spike_results = benchmark.stress_tester.spike_test(
    my_function,
    base_users=10,
    spike_users=100,
    spike_duration=30
)
```

## 📊 Data Visualization

### Rich Tables

```python
users = [
    {"name": "Alice", "age": 30, "city": "NYC"},
    {"name": "Bob", "age": 25, "city": "LA"},
    {"name": "Charlie", "age": 35, "city": "Chicago"}
]
log.table(users, title="User Data")
```

### Metrics Dashboard

```python
system_metrics = {
    "cpu_usage": 45.2,
    "memory_usage": 78.1,
    "requests_per_second": 1250,
    "error_rate": 0.02,
    "response_time_p95": 150
}
log.metrics(system_metrics, title="System Performance")
```

### JSON Pretty Printing

```python
api_response = {"users": [...], "pagination": {...}}
log.json(api_response, title="API Response")
```

## 🎯 Best Practices

### 1. **Structured Logging with Context**

```python
with log.context("user_registration") as ctx:
    with log.context("validation"):
        # validation logic
        pass

    with log.context("database"):
        # database operations
        pass

    log.success("Registration complete", user_id=new_user.id)
```

### 2. **Progressive Debugging**

```python
# Start simple
debug.ice(variable)

# Add tracing when needed
@trace.calls
def problematic_function():
    debug.ice("Function entry")
    # ... code ...
    debug.ice("Function exit")

# Add profiling for performance issues
@profile.comprehensive
@trace.snoop
def performance_critical():
    # ... code ...
```

### 3. **Production-Safe Usage**

```python
# Disable in production
if not settings.DEBUG:
    debug.disable()
    trace.call_tracker.disable()
    trace.var_tracker.disable()

# Or use environment-based configuration
DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"
if DEBUG_MODE:
    debug.enable()
    trace.call_tracker.enable()
```

## 🏗️ Architecture

The system is built with five core modules:

- **`debugging.py`** - Enhanced debugging utilities
- **`logging.py`** - Rich logging and output formatting
- **`tracing.py`** - Function and variable tracing
- **`profiling.py`** - Performance profiling and analysis
- **`benchmarking.py`** - Benchmarking and stress testing

All modules work independently or together, with smart fallbacks when optional dependencies are missing.

## 🧪 Testing

Comprehensive test suite with 400+ tests:

- **Unit tests** for all components
- **Integration tests** for workflows
- **Performance tests** for overhead
- **Error handling tests** for resilience

```bash
# Run all dev utilities tests
pytest tests/utils/test_dev_*.py -v

# Run specific module tests
pytest tests/utils/test_dev_debugging.py -v
pytest tests/utils/test_dev_logging.py -v
```

## 🔗 Dependencies

### Core (Always Available)

- **Python 3.8+**
- **Rich** - Beautiful terminal output (falls back to standard if missing)

### Optional (Graceful Fallbacks)

- **icecream** - Enhanced print debugging
- **pdb++** / **ipdb** - Enhanced debugger
- **web-pdb** - Web-based debugging
- **pudb** - Visual debugger
- **birdseye** - Call tracing
- **pysnooper** - Detailed tracing
- **hunter** - Advanced tracing
- **line_profiler** - Line-by-line profiling
- **memory_profiler** - Memory profiling
- **pyinstrument** - CPU profiling
- **scalene** - Advanced profiling
- **matplotlib** - Plotting and visualization
- **psutil** - System metrics

## 🚀 Getting Started

1. **Import and start using immediately:**

   ```python
   from haive.core.utils.dev import debug, log, trace, profile
   ```

2. **Add to your existing code gradually:**

   ```python
   # Replace print statements
   print(f"Value: {x}")  # Old
   debug.ice(x)          # New

   # Enhance logging
   logging.info("Done")  # Old
   log.success("Done!")  # New
   ```

3. **Profile performance bottlenecks:**

   ```python
   @profile.time
   def slow_function():
       # Your code here
       pass
   ```

4. **Debug complex issues:**
   ```python
   @trace.calls
   @debug.breakpoint_on_exception
   def complex_function():
       # Automatically debug on errors
       # Track all function calls
       pass
   ```

**🎯 Start simple, scale up as needed - the system grows with your debugging needs!**
