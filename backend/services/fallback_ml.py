import csv
import json
import math
import collections

# Simulated nan value
class NaNFinder:
    def __repr__(self):
        return 'nan'
    def __eq__(self, other):
        return isinstance(other, NaNFinder)
    def __ne__(self, other):
        return not isinstance(other, NaNFinder)

nan = NaNFinder()

def is_nan(val):
    if val is None:
        return True
    if isinstance(val, float) and math.isnan(val):
        return True
    if isinstance(val, NaNFinder):
        return True
    if str(val).lower() in ['nan', 'none', 'null', '']:
        return True
    return False

def isna(obj):
    if hasattr(obj, 'isna'):
        return obj.isna()
    return is_nan(obj)

class FallbackSeries:
    def __init__(self, data, name=None):
        self.data = list(data)
        self.name = name
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return self.data[idx]
        
    def __repr__(self):
        return f"FallbackSeries({self.data[:5]}..., name={self.name})"
        
    def isnull(self):
        return FallbackSeries([is_nan(x) for x in self.data], name=self.name)
        
    def sum(self):
        return sum(x for x in self.data if x is not None and not is_nan(x))
        
    def nunique(self):
        non_nulls = [x for x in self.data if not is_nan(x)]
        return len(set(non_nulls))
        
    @property
    def dtype(self):
        non_nulls = [x for x in self.data if not is_nan(x)]
        if not non_nulls:
            return "object"
        is_bool = all(isinstance(x, bool) or str(x).lower() in ['true', 'false'] for x in non_nulls)
        if is_bool:
            return "bool"
        try:
            all_int = all(int(x) == float(x) for x in non_nulls)
            if all_int:
                return "int64"
        except (ValueError, TypeError):
            pass
        try:
            [float(x) for x in non_nulls]
            return "float64"
        except (ValueError, TypeError):
            pass
        return "object"
        
    def min(self):
        clean = [float(x) for x in self.data if not is_nan(x)]
        return min(clean) if clean else 0.0
        
    def max(self):
        clean = [float(x) for x in self.data if not is_nan(x)]
        return max(clean) if clean else 0.0
        
    def mean(self):
        clean = [float(x) for x in self.data if not is_nan(x)]
        return sum(clean) / len(clean) if clean else 0.0
        
    def std(self):
        clean = [float(x) for x in self.data if not is_nan(x)]
        if len(clean) <= 1:
            return 0.0
        m = sum(clean) / len(clean)
        variance = sum((x - m) ** 2 for x in clean) / (len(clean) - 1)
        return math.sqrt(variance)
        
    def mode(self):
        clean = [str(x) for x in self.data if not is_nan(x)]
        if not clean:
            return FallbackSeries([])
        counter = collections.Counter(clean)
        most_common = counter.most_common(1)
        return FallbackSeries([most_common[0][0]] if most_common else [])
        
    @property
    def empty(self):
        return len(self.data) == 0
        
    @property
    def iloc(self):
        return self
        
    def fillna(self, value, inplace=False):
        new_data = [value if is_nan(x) else x for x in self.data]
        if inplace:
            self.data = new_data
            return self
        return FallbackSeries(new_data, name=self.name)
        
    def quantile(self, q):
        clean = sorted([float(x) for x in self.data if not is_nan(x)])
        if not clean:
            return 0.0
        k = (len(clean) - 1) * q
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return clean[int(k)]
        d0 = clean[int(f)] * (c - k)
        d1 = clean[int(c)] * (k - f)
        return d0 + d1
        
    def clip(self, lower, upper):
        new_data = []
        for x in self.data:
            if is_nan(x):
                new_data.append(x)
            else:
                try:
                    val = float(x)
                    new_data.append(max(lower, min(upper, val)))
                except (ValueError, TypeError):
                    new_data.append(x)
        return FallbackSeries(new_data, name=self.name)
        
    def astype(self, dtype_str):
        if dtype_str == 'str':
            return FallbackSeries([str(x) if x is not None else "" for x in self.data], name=self.name)
        return self

class FallbackDataFrame:
    def __init__(self, data_dict=None, columns=None):
        self._dict = data_dict or {}
        self.columns = columns or list(self._dict.keys())
        
    def __len__(self):
        if not self.columns:
            return 0
        return len(self._dict[self.columns[0]])
        
    def __getitem__(self, col):
        if isinstance(col, list):
            return FallbackDataFrame({c: self._dict[c] for c in col}, columns=col)
        if col not in self._dict:
            raise KeyError(f"Column '{col}' not found.")
        return FallbackSeries(self._dict[col], name=col)
        
    def __setitem__(self, col, val):
        if isinstance(val, FallbackSeries):
            self._dict[col] = val.data
        elif isinstance(val, FallbackDataFrame):
            for c in val.columns:
                self._dict[c] = val._dict[c]
        else:
            self._dict[col] = list(val)
        if col not in self.columns:
            self.columns.append(col)
            
    def copy(self):
        new_dict = {k: list(v) for k, v in self._dict.items()}
        return FallbackDataFrame(new_dict, columns=list(self.columns))
        
    def drop_duplicates(self, inplace=False):
        num_rows = len(self)
        if num_rows == 0:
            return self
        seen = set()
        unique_indices = []
        for i in range(num_rows):
            row_tuple = tuple(self._dict[col][i] for col in self.columns)
            if row_tuple not in seen:
                seen.add(row_tuple)
                unique_indices.append(i)
                
        new_dict = {col: [self._dict[col][i] for i in unique_indices] for col in self.columns}
        if inplace:
            self._dict = new_dict
            return self
        return FallbackDataFrame(new_dict, columns=list(self.columns))
        
    def head(self, n=10):
        new_dict = {col: self._dict[col][:n] for col in self.columns}
        return FallbackDataFrame(new_dict, columns=list(self.columns))
        
    def replace(self, replace_dict):
        new_dict = {}
        for col in self.columns:
            new_col = []
            for x in self._dict[col]:
                replaced = False
                for k, v in replace_dict.items():
                    if is_nan(x):
                        new_col.append(v)
                        replaced = True
                        break
                if not replaced:
                    new_col.append(x)
            new_dict[col] = new_col
        return FallbackDataFrame(new_dict, columns=list(self.columns))
        
    def to_dict(self, orient="records"):
        if orient != "records":
            raise ValueError("Only 'records' orient is supported in fallback")
        num_rows = len(self)
        records = []
        for i in range(num_rows):
            row = {}
            for col in self.columns:
                val = self._dict[col][i]
                if is_nan(val):
                    row[col] = None
                else:
                    row[col] = val
            records.append(row)
        return records

def read_csv(filepath_or_buffer):
    with open(filepath_or_buffer, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        try:
            columns = next(reader)
        except StopIteration:
            return FallbackDataFrame()
            
        data_dict = {col: [] for col in columns}
        for row in reader:
            if not row:
                continue
            for idx, col in enumerate(columns):
                val = row[idx] if idx < len(row) else ""
                try:
                    if '.' in val:
                        parsed = float(val)
                    else:
                        parsed = int(val)
                except ValueError:
                    if val.lower() in ['true', 'false']:
                        parsed = val.lower() == 'true'
                    elif val.lower() in ['nan', 'none', 'null', '']:
                        parsed = nan
                    else:
                        parsed = val
                data_dict[col].append(parsed)
        return FallbackDataFrame(data_dict, columns=columns)

def read_json(filepath_or_buffer):
    with open(filepath_or_buffer, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, list):
        if not data:
            return FallbackDataFrame()
        columns = list(data[0].keys())
        data_dict = {col: [] for col in columns}
        for row in data:
            for col in columns:
                val = row.get(col, nan)
                data_dict[col].append(val)
        return FallbackDataFrame(data_dict, columns=columns)
    elif isinstance(data, dict):
        columns = list(data.keys())
        first_val = data[columns[0]] if columns else None
        if isinstance(first_val, dict):
            indices = list(first_val.keys())
            data_dict = {col: [data[col].get(idx, nan) for idx in indices] for col in columns}
            return FallbackDataFrame(data_dict, columns=columns)
        else:
            data_dict = {col: list(data[col]) for col in columns}
            return FallbackDataFrame(data_dict, columns=columns)
    return FallbackDataFrame()

def read_excel(filepath_or_buffer):
    raise ValueError("Excel file format is not supported in zero-config fallback. Please upload a CSV instead.")

class PandasFallback:
    def __init__(self):
        self.nan = nan
        self.DataFrame = FallbackDataFrame
        self.Series = FallbackSeries
        self.read_csv = read_csv
        self.read_json = read_json
        self.read_excel = read_excel
        
    def isna(self, obj):
        return isna(obj)

class NumpyFallback:
    def __init__(self):
        self.nan = nan
        self.integer = int
        self.floating = float
        
    def issubdtype(self, dtype1, dtype2):
        if dtype2 == "integer" or dtype2 == int or dtype2 == "int64":
            return "int" in str(dtype1)
        if dtype2 == "floating" or dtype2 == float or dtype2 == "float64":
            return "float" in str(dtype1)
        return False
        
    def linspace(self, start, stop, num=50):
        if num <= 1:
            return [start]
        step = (stop - start) / (num - 1)
        return [start + i * step for i in range(num)]
        
    def isnan(self, val):
        return is_nan(val)

class FallbackStandardScaler:
    def fit_transform(self, df):
        new_dict = {}
        for col in df.columns:
            series = df[col]
            m = series.mean()
            s = series.std()
            if s == 0:
                s = 1.0
            new_dict[col] = [(float(x) - m) / s for x in series.data]
        return FallbackDataFrame(new_dict, columns=list(df.columns))

class FallbackMinMaxScaler:
    def fit_transform(self, df):
        new_dict = {}
        for col in df.columns:
            series = df[col]
            low = series.min()
            high = series.max()
            diff = high - low
            if diff == 0:
                diff = 1.0
            new_dict[col] = [(float(x) - low) / diff for x in series.data]
        return FallbackDataFrame(new_dict, columns=list(df.columns))

class FallbackLabelEncoder:
    def fit_transform(self, y):
        data = list(y)
        unique_sorted = sorted(list(set(str(x) for x in data)))
        label_map = {val: idx for idx, val in enumerate(unique_sorted)}
        return [label_map[str(x)] for x in data]

pd = PandasFallback()
np = NumpyFallback()
