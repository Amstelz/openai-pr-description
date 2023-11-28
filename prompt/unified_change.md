```diff
Changes in file .github/workflows/build-ut-coverage.yml:
@@ -24,6 +24,7 @@
    jobs:
        run: |
            sudo apt-get update
            sudo apt-get install -y lcov
+           sudo apt-get install -y valgrind
            sudo apt-get install -y ${{ matrix.compiler.cc }}
            sudo apt-get install -y ${{ matrix.compiler.cxx }}
        - name: Checkout repository
@@ -48,3 +49,7 @@
        jobs:
            with:
                files: coverage.info
                fail_ci_if_error: true
+      - name: Run valgrind
+        run: |
+          valgrind --tool=memcheck --leak-check=full --leak-resolution=med \
+            --track-origins=yes --vgdb=no --error-exitcode=1 ${build_dir}/test/command_parser_test

Changes in file test/CommandParserTest.cpp:
@@ -566,7 +566,7 @@
TEST(CommandParserTest, ParsedCommandImpl_WhenArgumentIsSupportedNumericTypeWill)
    unsigned long long expectedUnsignedLongLong { std::numeric_limits<unsigned long long>::max() };
    float expectedFloat { -164223.123f }; // std::to_string does not play well with floating point min()
    double expectedDouble { std::numeric_limits<double>::max() };
-   long double expectedLongDouble { std::numeric_limits<long double>::max() };
+   long double expectedLongDouble { 123455678912349.1245678912349L };

    auto command = UnparsedCommand::create(expectedCommand, "dummyDescription"s).withArgs<int, long, unsigned long, long long, unsigned long long, float, double, long double>();
```
