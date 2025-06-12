# Troubleshooting Guide & Common Issues

This page documents common errors encountered during the development of this project and their solutions. It serves as a reference for debugging and understanding potential pitfalls when working with PySpark, Delta Lake, and data pipelines on a local machine.

1. PySpark: java.net.URISyntaxException or IllegalArgumentException on Windows


Symptom: The Spark application fails immediately upon initialization. The error message contains something like java.net.URISyntaxException: Relative path in absolute URI or points to a malformed path.



Root Cause: This is a classic issue on Windows when your project path contains spaces. Spark's underlying Java components often fail to parse these paths correctly if they are not properly quoted. For example, a path like D:\My Project\spark-warehouse will be misinterpreted.

Solution:

(Recommended) Move the entire project directory to a path without spaces (e.g., D:\projects\ecommerce-recsys). This is the most robust solution and a general best practice for development.

(Workaround) If moving the project is not an option, you can try explicitly quoting the paths in your Spark configuration, but this can be brittle.


*In create_spark_session()*
```
.config("spark.sql.warehouse.dir", 'file:///D:/My%20Project/spark-warehouse') # URL-encode spaces
```



2. PySpark: AttributeError: 'get' on pyspark.sql.Row objects


Symptom: The script fails inside a standard Python function (not a UDF) that processes data previously collected from a DataFrame. The traceback points to a line using row.get("column_name") and ends with AttributeError: get.




Root Cause: The .collect() action on a Spark DataFrame returns a list of pyspark.sql.Row objects, not a list of standard Python dictionaries. While Row objects are convenient, they do not have a .get() method. You must access their elements using attribute-style access (row.column_name) or index-style access (row['column_name']).




Solution:




(Direct Access) Modify the function to use attribute access. This is clean and efficient.


*Instead of this:*
```
value = row.get("my_column")
```

*Do this:*
```
value = row.my_column
```



(Convert to Dictionary) If the function needs to be more generic, convert the Row objects to dictionaries immediately after collecting them.


*Collect and convert*
```
data_list = [row.asDict() for row in my_dataframe.collect()]
```

*Now your function can safely use .get()*
def my_function(data):
    for row in data:
        value = row.get("my_column")





3. Delta Lake: AnalysisException: A schema mismatch detected


Symptom: The script fails during a df.write.format("delta").mode("overwrite").save(path) operation. The error message explicitly states a schema mismatch between the existing Delta table and the new data you are trying to write.




Root Cause: Delta Lake, by default, enforces schema to prevent accidental data corruption. When using mode("overwrite"), it expects the new data to have the exact same schema (column names, types, and order) as the data it's replacing. If you've changed a column name, added a column, or altered a data type, this safety check will trigger an error.




Solution:



If the schema change is intentional and you want to completely replace the old table structure, you must explicitly allow Spark to overwrite the schema. Add the .option("overwriteSchema", "true") to your write operation.

```
(my_dataframe.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true") # This tells Delta it's OK to change the schema
    .save(table_path))
```


4. PySpark: Py4JJavaError: ... java.io.FileNotFoundException when reading data


Symptom: Spark fails to read a CSV or Parquet file, throwing a FileNotFoundException, even though you can see the file exists in your file explorer.




Root Cause: This is often a working directory issue. When you run a Python script, its "current working directory" might not be what you expect, especially if you run it from a different location or within an IDE. Spark tries to resolve relative paths like ./data/mart from this working directory.




Solution:



(Best Practice) Use absolute paths constructed dynamically. This makes the script runnable from anywhere.
import pathlib

*Get the root directory of the project*

PROJECT_ROOT = pathlib.Path(__file__).parent.resolve()

*Construct path to data*

DATA_PATH = PROJECT_ROOT.joinpath("data", "my_file.csv")

*Now use this absolute path*
df = spark.read.csv(str(DATA_PATH))


(Quick Fix) Ensure you are running the python your_script.py command from the root directory of your project.



5. Environment: HADOOP_HOME not set or winutils.exe not found


Symptom: Spark fails to start on Windows, often with a NullPointerException or an error message related to file system access.


Root Cause: PySpark on Windows requires a helper binary called winutils.exe and some associated Hadoop DLLs to interact with the local file system correctly. This is not included with Spark and must be downloaded separately.




Solution:



Download winutils.exe: Find a winutils.exe binary that matches the version of Hadoop your Spark distribution was built for (e.g., Hadoop 3.3 for Spark 3.4). A common repository is steveloughran/winutils.

Create a hadoop directory: Create a folder, for example, C:\hadoop.

Create a bin subdirectory: Inside 
```
C:\hadoop
```
create a bin folder (C:\hadoop\bin).

Place winutils.exe: Copy the downloaded winutils.exe into C:\hadoop\bin.

Set Environment Variable: Set the HADOOP_HOME environment variable to point to C:\hadoop.

*In your system's environment variables or in your script*
os.environ['HADOOP_HOME'] = r'C:\hadoop'





This guide helps in quickly diagnosing and resolving the most frequent issues, ensuring a smoother development and execution workflow.
